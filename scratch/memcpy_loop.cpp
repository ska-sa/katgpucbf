/*******************************************************************************
 * Copyright (c) 2019-2020, 2022-2024, National Research Foundation (SARAO)
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy
 * of the License at
 *
 *   https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

/**
 * Memory copy micro-benchmark. It is designed to test both the memory
 * thoughput of a system and of various subsets of cores, as well as to
 * test various methods of performing a copy.
 *
 * The command-line takes a list of cores on which to run copies, as well
 * as the following options:
 *
 * -t: memory allocation method (malloc/mmap/mmap_huge/madv_huge)
 * -f: memory copy function (see below)
 * -p: number of times to do a copy before printing a rate
 * -r: number of times to run -p passes and print a rate (default is infinite)
 * -b: size of the buffer to copy
 * -c: size of individual calls to copy function
 * -S: an offset to add to the source address
 * -D: an offset to add to the destination address
 * -T: run tests of the function implementations
 *
 * The supported functions are:
 *
 * - memcpy: the library memcpy implementation
 * - memcpy_sse2/avx/avx512: x86 SIMD copies
 * - memcpy_stream_sse2/avx/avx512: x86 SIMD copies, using streaming stores
 * - memcpy_rep_movsb: use the x86 "REP MOVSB" instruction
 * - memcpy_sve: ARM SIMD copy
 * - memcpy_stream_sve: ARM SIMD copy, using streaming loads and stores
 * - memcpy_*_reverse: variants that copy from highest address to lowest
 * - memset: use library memset to clear the destination
 * - memset_stream_sse2: use SSE2 streaming stores to clear the destination
 * - memset_stream_sve: use SVE streaming stores to clear the destination
 * - read: just read the source (using SSE2) and do not write anything
 */

#include <iostream>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <chrono>
#include <future>
#include <memory>
#include <random>
#include <algorithm>

#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>
#ifdef __x86_64__
# include <emmintrin.h>
# include <immintrin.h>
#endif
#ifdef __ARM_FEATURE_SVE
# include <sys/auxv.h>
# include <arm_sve.h>
# include <arm_acle.h>
#endif

using namespace std;
using namespace std::chrono;
using namespace std::literals::string_literals;

static constexpr size_t cache_line_size = 64;  // guesstimate

enum class memory_type
{
    MALLOC,
    MMAP,
    MMAP_HUGE,
    MADV_HUGE
};

enum class memory_function
{
    MEMCPY,
#ifdef __x86_64__
    MEMCPY_SSE2,
    MEMCPY_SSE2_REVERSE,
    MEMCPY_AVX,
    MEMCPY_AVX_REVERSE,
    MEMCPY_AVX512,
    MEMCPY_AVX512_REVERSE,
    MEMCPY_STREAM_SSE2,
    MEMCPY_STREAM_SSE2_REVERSE,
    MEMCPY_STREAM_AVX,
    MEMCPY_STREAM_AVX_REVERSE,
    MEMCPY_STREAM_AVX512,
    MEMCPY_STREAM_AVX512_REVERSE,
    MEMCPY_REP_MOVSB,
    MEMCPY_REP_MOVSB_REVERSE,
#endif // __x86_64__
#ifdef __ARM_FEATURE_SVE
    MEMCPY_SVE,
    MEMCPY_STREAM_SVE,
#endif // __ARM_FEATURE_SVE
    MEMSET,
#ifdef __x86_64__
    MEMSET_STREAM_SSE2,
    READ,
#endif // __x86_64__
#ifdef __ARM_FEATURE_SVE
    MEMSET_STREAM_SVE,
#endif
};

enum class memory_function_type
{
    MEMCPY,
    MEMSET,
    READ
};

static char *allocate(size_t size, memory_type type)
{
    void *addr;
    if (type == memory_type::MALLOC)
    {
        // Ensure at least cache line alignment
        size_t space = size + cache_line_size;
        addr = malloc(size + cache_line_size);
        if (addr == nullptr)
            throw bad_alloc();
        align(cache_line_size, size, addr, space);
    }
    else
    {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (type == memory_type::MMAP_HUGE)
            flags |= MAP_HUGETLB;
        addr = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (addr == MAP_FAILED)
            throw bad_alloc();
        if (type == memory_type::MADV_HUGE)
            madvise(addr, size, MADV_HUGEPAGE);
    }
    memset(addr, 1, size);  // ensure it has real pages
    return (char *) addr;
}

/**
 * Template for memcpy implementations. The copy uses elements of type V, which
 * are loaded with L and stored with S. The main loop is unrolled by a factor
 * of unroll1, and the tail is handled with a loop unrolled by unroll2 (should
 * divide into unroll1). At the end, F is called.
 *
 * This is intended to be wrapped by memcpy_aligned, since it does no internal
 * alignment, and requires n to be a multiple of unroll2 elements.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"  // GCC warns that it might not be inlinable
template<typename V, int unroll1, int unroll2, int alignment, typename L, typename S, typename F>
[[gnu::always_inline]]
static void *memcpy_generic(
    void * __restrict__ dest, const void * __restrict__ src, size_t n,
    const L &load, const S &store, const F &fence) noexcept
{
    static_assert(unroll1 % unroll2 == 0, "unroll1 must be a multiple of unroll2");
    static_assert(alignment > 0 && (alignment & (alignment - 1)) == 0, "unalignment must be a power of 2");

    void *aligned_dest = dest;
    const void *aligned_src = src;  // not necessarily aligned, but corresponds to aligned_dest
    if (!align(alignment, unroll2 * sizeof(V), aligned_dest, n))
    {
        // Not even room for one aligned block. Just fall back to plain memcpy
        return std::memcpy(dest, src, n);
    }
    // Copy the head
    size_t head = (char *) aligned_dest - (char *) dest;
    if (head != 0)
    {
        std::memcpy(dest, src, head);
        aligned_src = (const void *) ((const char *) src + head);
    }

    char * __restrict__ dest_c = (char *) aligned_dest;
    const char * __restrict__ src_c = (const char *) aligned_src;
    size_t offset = 0;
    for (; offset + unroll1 * sizeof(V) <= n; offset += unroll1 * sizeof(V))
    {
        V values[unroll1];
        for (int i = 0; i < unroll1; i++)
            load((V const *) (src_c + offset + i * sizeof(V)), values[i]);
        for (int i = 0; i < unroll1; i++)
            store((V *) (dest_c + offset + i * sizeof(V)), values[i]);
    }
    if constexpr (unroll2 != unroll1)
    {
        for (; offset + unroll2 * sizeof(V) <= n; offset += unroll2 * sizeof(V))
        {
            V values[unroll2];
            for (int i = 0; i < unroll2; i++)
                load((V const *) (src_c + offset + i * sizeof(V)), values[i]);
            for (int i = 0; i < unroll2; i++)
                store((V *) (dest_c + offset + i * sizeof(V)), values[i]);
        }
    }
    fence();

    size_t tail = n - offset;
    if (tail != 0)
    {
        std::memcpy(dest_c + offset, src_c + offset, tail);
    }

    return dest;
}

/* Similar to memcpy_generic, but runs backwards */
template<typename V, int unroll1, int unroll2, int alignment, typename L, typename S, typename F>
[[gnu::always_inline]]
static void *memcpy_generic_reverse(
    void * __restrict__ dest, const void * __restrict__ src, size_t n,
    const L &load, const S &store, const F &fence) noexcept
{
    static_assert(unroll1 % unroll2 == 0, "unroll1 must be a multiple of unroll2");
    static_assert(alignment > 0 && (alignment & (alignment - 1)) == 0, "unalignment must be a power of 2");
    constexpr size_t block1 = unroll1 * sizeof(V);
    constexpr size_t block2 = unroll2 * sizeof(V);

    void *aligned_dest = dest;
    const void *aligned_src = src;  // not necessarily aligned, but corresponds to aligned_dest
    if (!align(alignment, block2, aligned_dest, n))
    {
        // Not even room for one aligned block. Just fall back to plain memcpy
        return std::memcpy(dest, src, n);
    }
    size_t head = (char *) aligned_dest - (char *) dest;
    if (head != 0)
        aligned_src = (const void *) ((const char *) src + head);

    char * __restrict__ dest_c = (char *) aligned_dest;
    const char * __restrict__ src_c = (const char *) aligned_src;

    size_t bulk_n = n / block2 * block2;
    size_t tail = n - bulk_n;
    if (tail != 0)
    {
        std::memcpy(dest_c + bulk_n, src_c + bulk_n, tail);
    }

    size_t offset = bulk_n;
    if constexpr (unroll2 != unroll1)
    {
        while (offset % block1 != 0)
        {
            V values[unroll2];
            offset -= block2;
            for (int i = unroll2 - 1; i >= 0; i--)
                load((V const *) (src_c + offset + i * sizeof(V)), values[i]);
            for (int i = unroll2 - 1; i >= 0; i--)
                store((V *) (dest_c + offset + i * sizeof(V)), values[i]);
        }
    }
    while (offset != 0)
    {
        V values[unroll1];
        offset -= block1;
        for (int i = unroll1 - 1; i >= 0; i--)
            load((V const *) (src_c + offset + i * sizeof(V)), values[i]);
        for (int i = unroll1 - 1; i >= 0; i--)
            store((V *) (dest_c + offset + i * sizeof(V)), values[i]);
    }

    // Copy the head
    if (head != 0)
        std::memcpy(dest, src, head);

    fence();
    return dest;
}
#pragma GCC diagnostic pop

#if __x86_64__
// memcpy, with SSE2
static void *memcpy_sse2(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic<__m128i, 4, 4, cache_line_size>(
        dest, src, n,
        [](const __m128i *ptr, __m128i &value) { value = _mm_loadu_si128(ptr); },
        [](__m128i *ptr, __m128i value) { _mm_store_si128(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with SSE2, reversed
static void *memcpy_sse2_reverse(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic_reverse<__m128i, 4, 4, cache_line_size>(
        dest, src, n,
        [](const __m128i *ptr, __m128i &value) { value = _mm_loadu_si128(ptr); },
        [](__m128i *ptr, __m128i value) { _mm_store_si128(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with SSE2 streaming stores
static void *memcpy_stream_sse2(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic<__m128i, 4, 4, cache_line_size>(
        dest, src, n,
        [](const __m128i *ptr, __m128i &value) { value = _mm_loadu_si128(ptr); },
        [](__m128i *ptr, __m128i value) { _mm_stream_si128(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with SSE2 streaming stores, reversed
static void *memcpy_stream_sse2_reverse(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic_reverse<__m128i, 4, 4, cache_line_size>(
        dest, src, n,
        [](const __m128i *ptr, __m128i &value) { value = _mm_loadu_si128(ptr); },
        [](__m128i *ptr, __m128i value) { _mm_stream_si128(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with AVX
[[gnu::target("avx")]]
static void *memcpy_avx(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic<__m256i, 8, 2, cache_line_size>(
        dest, src, n,
        [] [[gnu::target("avx")]] (const __m256i *ptr, __m256i &value) { value = _mm256_loadu_si256(ptr); },
        [] [[gnu::target("avx")]] (__m256i *ptr, __m256i value) { _mm256_store_si256(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with AVX, reversed
[[gnu::target("avx")]]
static void *memcpy_avx_reverse(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic_reverse<__m256i, 8, 2, cache_line_size>(
        dest, src, n,
        [] [[gnu::target("avx")]] (const __m256i *ptr, __m256i &value) { value = _mm256_loadu_si256(ptr); },
        [] [[gnu::target("avx")]] (__m256i *ptr, __m256i value) { _mm256_store_si256(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with AVX streaming stores
[[gnu::target("avx")]]
static void *memcpy_stream_avx(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic<__m256i, 8, 2, cache_line_size>(
        dest, src, n,
        [] [[gnu::target("avx")]] (const __m256i *ptr, __m256i &value) { value = _mm256_loadu_si256(ptr); },
        [] [[gnu::target("avx")]] (__m256i *ptr, __m256i value) { _mm256_stream_si256(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with AVX streaming stores, reversed
[[gnu::target("avx")]]
static void *memcpy_stream_avx_reverse(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic_reverse<__m256i, 8, 2, cache_line_size>(
        dest, src, n,
        [] [[gnu::target("avx")]] (const __m256i *ptr, __m256i &value) { value = _mm256_loadu_si256(ptr); },
        [] [[gnu::target("avx")]] (__m256i *ptr, __m256i value) { _mm256_stream_si256(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with AVX-512
[[gnu::target("avx512f")]]
static void *memcpy_avx512(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic<__m512i, 8, 1, cache_line_size>(
        dest, src, n,
        [] [[gnu::target("avx512f")]] (const __m512i *ptr, __m512i &value) { value = _mm512_loadu_si512(ptr); },
        [] [[gnu::target("avx512f")]] (__m512i *ptr, __m512i value) { _mm512_store_si512(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with AVX-512, reversed
[[gnu::target("avx512f")]]
static void *memcpy_avx512_reverse(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic_reverse<__m512i, 8, 1, cache_line_size>(
        dest, src, n,
        [] [[gnu::target("avx512f")]] (const __m512i *ptr, __m512i &value) { value = _mm512_loadu_si512(ptr); },
        [] [[gnu::target("avx512f")]] (__m512i *ptr, __m512i value) { _mm512_store_si512(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with AVX-512 streaming stores
[[gnu::target("avx512f")]]
static void *memcpy_stream_avx512(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic<__m512i, 8, 1, cache_line_size>(
        dest, src, n,
        [] [[gnu::target("avx512f")]] (const __m512i *ptr, __m512i &value) { value = _mm512_loadu_si512(ptr); },
        [] [[gnu::target("avx512f")]] (__m512i *ptr, __m512i value) { _mm512_stream_si512(ptr, value); },
        []() { _mm_sfence(); }
    );
}

// memcpy, with AVX-512 streaming stores, reversed
[[gnu::target("avx512f")]]
static void *memcpy_stream_avx512_reverse(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_generic_reverse<__m512i, 8, 1, cache_line_size>(
        dest, src, n,
        [] [[gnu::target("avx512f")]] (const __m512i *ptr, __m512i &value) { value = _mm512_loadu_si512(ptr); },
        [] [[gnu::target("avx512f")]] (__m512i *ptr, __m512i value) { _mm512_stream_si512(ptr, value); },
        []() { _mm_sfence(); }
    );
}

static void *memcpy_rep_movsb(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    void *orig_dest = dest;
    asm volatile("rep movsb" : "+c" (n), "+D" (dest), "+S" (src) : : "memory");
    return orig_dest;
}

static void *memcpy_rep_movsb_reverse(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    void *orig_dest = dest;
    dest = (char *) dest + (n - 1);
    src = (const char *) src + (n - 1);
    asm volatile("std; rep movsb; cld" : "+c" (n), "+D" (dest), "+S" (src) : : "memory");
    return orig_dest;
}

/* memset, but using SSE streaming stores */
static void *memset_stream_sse2(void *dst, int c, size_t bytes) noexcept
{
    // Simplifies some edge cases
    if (bytes <= 16)
    {
        return std::memset(dst, c, bytes);
    }

    // Process prefix up to 16-byte alignment
    char *cdst = (char *) dst;
    char *cdst_round = (char *) ((uintptr_t(dst) + 0xf) & ~0xf);
    if (cdst != cdst_round)
    {
        std::memset(dst, c, cdst_round - cdst);
        bytes -= cdst_round - cdst;
    }

    // Use streaming stores for the bulk
    __m128i value;
    std::memset(&value, c, sizeof(value));
    __m128i *mdst = (__m128i *) cdst_round;
    __m128i *mend = mdst + (bytes / 16);
    bytes -= 16 * (mend - mdst);
    while (mdst != mend)
    {
        _mm_stream_si128(mdst, value);
        mdst++;
    }
    _mm_sfence();

    // Handle suffix
    if (bytes > 0)
        std::memset(mdst, c, bytes);
    return dst;
}

/* Read all the data in [src, src + bytes) and do nothing with it. */
static void memory_read(const void *src, size_t bytes) noexcept
{
    uint8_t result1 = 0;
    // Process prefix up to 16-byte alignment
    const char *csrc = (const char *) src;
    while (((uintptr_t) csrc) & 0xf)
    {
        result1 ^= *csrc++;
        bytes--;
        if (bytes == 0)
            break;
    }

    // Process main body
    __m128i result2 = _mm_setzero_si128();
    const __m128i *msrc = (const __m128i *) csrc;
    const __m128i *mend = msrc + (bytes / 16);
    bytes -= 16 * (mend - msrc);
    while (msrc != mend)
    {
        result2 = _mm_xor_si128(result2, _mm_load_si128(msrc));
        msrc++;
    }

    // Process tail
    csrc = (const char *) msrc;
    while (bytes > 0)
    {
        result1 ^= *csrc++;
        bytes--;
    }

    /* Dump the results into volatile variables to prevent the compiler
     * optimising the whole thing away.
     */
    volatile uint8_t sink1 = result1;
    volatile __m128i sink2 = result2;
    // Suppress unused variable warnings
    (void) sink1;
    (void) sink2;
}
#endif // __x86_64__

#ifdef __ARM_FEATURE_SVE

template<typename L, typename S>
static void *memcpy_sve_generic(
    void * __restrict__ dest, const void * __restrict__ src, size_t n,
    const L &load, const S &store) noexcept
{
    std::uint8_t *destc = (std::uint8_t *) dest;
    const std::uint8_t *srcc = (const std::uint8_t *) src;
    const size_t vsize = svcntb();
    static constexpr int unroll = 2; // keep in sync with actual code

    /* Experiments on Grace (Neoverse V2) show that source alignment
     * is more important than destination alignment to throughput.
     */
    void *aligned_src = const_cast<void *>(src);
    if (align(vsize, vsize * unroll, aligned_src, n))
    {
        std::size_t head = (const std::uint8_t *) aligned_src - srcc;
        svbool_t pg = svwhilelt_b8(std::size_t(0), head);
        store(pg, destc, load(pg, srcc));
        destc += head;
        srcc += head;
    }

    size_t i = 0;
    while (i + unroll * vsize <= n)
    {
        // Unfortunately svuint8_t is a "sizeless" type, which can't be
        // put into an array. So we have to hand-unroll.
        svuint8_t data0, data1;
        data0 = load(svptrue_b8(), &srcc[i + 0 * vsize]);
        data1 = load(svptrue_b8(), &srcc[i + 1 * vsize]);
        store(svptrue_b8(), &destc[i + 0 * vsize], data0);
        store(svptrue_b8(), &destc[i + 1 * vsize], data1);
        i += unroll * vsize;
    }
    svbool_t pg = svwhilelt_b8(i, n);
    do
    {
        store(pg, &destc[i], load(pg, &srcc[i]));
        i += vsize;
    } while (svptest_first(svptrue_b8(), pg = svwhilelt_b8(i, n)));
    return dest;
}

static void *memcpy_sve(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_sve_generic(
        dest, src, n,
        [](svbool_t pred, const std::uint8_t *ptr) { return svld1_u8(pred, ptr); },
        [](svbool_t pred, std::uint8_t *ptr, svuint8_t value) { return svst1_u8(pred, ptr, value); }
    );
}

static void *memcpy_stream_sve(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_sve_generic(
        dest, src, n,
        [](svbool_t pred, const std::uint8_t *ptr) { return svldnt1_u8(pred, ptr); },
        [](svbool_t pred, std::uint8_t *ptr, svuint8_t value) { return svstnt1_u8(pred, ptr, value); }
    );
}

static void *memset_stream_sve(void *dest, int c, size_t n) noexcept
{
    std::uint8_t *destc = (std::uint8_t *) dest;
    const size_t vsize = svcntb();
    static constexpr int unroll = 2; // keep in sync with actual code

    void *aligned_dest = const_cast<void *>(dest);
    svuint8_t value = svdup_u8(c);
    if (align(vsize, vsize * unroll, aligned_dest, n))
    {
        std::size_t head = (const std::uint8_t *) aligned_dest - destc;
        svbool_t pg = svwhilelt_b8(std::size_t(0), head);
        svstnt1_u8(pg, destc, value);
        destc += head;
    }

    size_t i = 0;
    while (i + unroll * vsize <= n)
    {
        // Unfortunately svuint8_t is a "sizeless" type, which can't be
        // put into an array. So we have to hand-unroll.
        svstnt1_u8(svptrue_b8(), &destc[i + 0 * vsize], value);
        svstnt1_u8(svptrue_b8(), &destc[i + 1 * vsize], value);
        i += unroll * vsize;
    }
    svbool_t pg = svwhilelt_b8(i, n);
    do
    {
        svstnt1_u8(pg, &destc[i], value);
        i += vsize;
    } while (svptest_first(svptrue_b8(), pg = svwhilelt_b8(i, n)));
    return dest;
}

static bool sve_supported()
{
    return getauxval(AT_HWCAP) & HWCAP_SVE;
}

#endif // __ARM_FEATURE_SVE

static const struct
{
    memory_type value;
    string name;
} memory_types[] = {
    { memory_type::MALLOC, "malloc"s },
    { memory_type::MMAP, "mmap"s },
    { memory_type::MMAP_HUGE, "mmap_huge"s },
    { memory_type::MADV_HUGE, "madv_huge"s },
};

static const struct
{
    memory_function value;
    memory_function_type type;
    string name;
    bool supported;
    union
    {
        void *(*memcpy_impl)(void * __restrict__, const void * __restrict__, size_t) noexcept;
        void *(*memset_impl)(void *, int, size_t) noexcept;
        void (*read_impl)(const void *, size_t) noexcept;
    } impl;
} memory_functions[] = {
    {
        memory_function::MEMCPY,
        memory_function_type::MEMCPY,
        "memcpy",
        true,
        { .memcpy_impl = &std::memcpy },
    },
#if __x86_64__
    {
        memory_function::MEMCPY_SSE2,
        memory_function_type::MEMCPY,
        "memcpy_sse2",
        bool(__builtin_cpu_supports("sse2")),
        { .memcpy_impl = &memcpy_sse2 },
    },
    {
        memory_function::MEMCPY_SSE2_REVERSE,
        memory_function_type::MEMCPY,
        "memcpy_sse2_reverse",
        bool(__builtin_cpu_supports("sse2")),
        { .memcpy_impl = &memcpy_sse2_reverse },
    },
    {
        memory_function::MEMCPY_AVX,
        memory_function_type::MEMCPY,
        "memcpy_avx",
        bool(__builtin_cpu_supports("avx")),
        { .memcpy_impl = &memcpy_avx },
    },
    {
        memory_function::MEMCPY_AVX_REVERSE,
        memory_function_type::MEMCPY,
        "memcpy_avx_reverse",
        bool(__builtin_cpu_supports("avx")),
        { .memcpy_impl = &memcpy_avx_reverse },
    },
    {
        memory_function::MEMCPY_AVX512,
        memory_function_type::MEMCPY,
        "memcpy_avx512",
        bool(__builtin_cpu_supports("avx512f")),
        { .memcpy_impl = &memcpy_avx512 },
    },
    {
        memory_function::MEMCPY_AVX512_REVERSE,
        memory_function_type::MEMCPY,
        "memcpy_avx512_reverse",
        bool(__builtin_cpu_supports("avx512f")),
        { .memcpy_impl = &memcpy_avx512_reverse },
    },
    {
        memory_function::MEMCPY_STREAM_SSE2,
        memory_function_type::MEMCPY,
        "memcpy_stream_sse2",
        bool(__builtin_cpu_supports("sse2")),
        { .memcpy_impl = &memcpy_stream_sse2 },
    },
    {
        memory_function::MEMCPY_STREAM_SSE2_REVERSE,
        memory_function_type::MEMCPY,
        "memcpy_stream_sse2_reverse",
        bool(__builtin_cpu_supports("sse2")),
        { .memcpy_impl = &memcpy_stream_sse2_reverse },
    },
    {
        memory_function::MEMCPY_STREAM_AVX,
        memory_function_type::MEMCPY,
        "memcpy_stream_avx",
        bool(__builtin_cpu_supports("avx")),
        { .memcpy_impl = &memcpy_stream_avx },
    },
    {
        memory_function::MEMCPY_STREAM_AVX_REVERSE,
        memory_function_type::MEMCPY,
        "memcpy_stream_avx_reverse",
        bool(__builtin_cpu_supports("avx")),
        { .memcpy_impl = &memcpy_stream_avx_reverse },
    },
    {
        memory_function::MEMCPY_STREAM_AVX512,
        memory_function_type::MEMCPY,
        "memcpy_stream_avx512",
        bool(__builtin_cpu_supports("avx512f")),
        { .memcpy_impl = &memcpy_stream_avx512 },
    },
    {
        memory_function::MEMCPY_STREAM_AVX512_REVERSE,
        memory_function_type::MEMCPY,
        "memcpy_stream_avx512_reverse",
        bool(__builtin_cpu_supports("avx512f")),
        { .memcpy_impl = &memcpy_stream_avx512_reverse },
    },
    {
        memory_function::MEMCPY_REP_MOVSB,
        memory_function_type::MEMCPY,
        "memcpy_rep_movsb",
        true,
        { .memcpy_impl = &memcpy_rep_movsb },
    },
    {
        memory_function::MEMCPY_REP_MOVSB_REVERSE,
        memory_function_type::MEMCPY,
        "memcpy_rep_movsb_reverse",
        true,
        { .memcpy_impl = &memcpy_rep_movsb_reverse },
    },
#endif // __x86_64__
#ifdef __ARM_FEATURE_SVE
    {
        memory_function::MEMCPY_SVE,
        memory_function_type::MEMCPY,
        "memcpy_sve",
        sve_supported(),
        { .memcpy_impl = &memcpy_sve },
    },
    {
        memory_function::MEMCPY_STREAM_SVE,
        memory_function_type::MEMCPY,
        "memcpy_stream_sve",
        sve_supported(),
        { .memcpy_impl = &memcpy_stream_sve },
    },
#endif // __ARM_FEATURE_SVE
    {
        memory_function::MEMSET,
        memory_function_type::MEMSET,
        "memset",
        true,
        { .memset_impl = &std::memset },
    },
#ifdef __x86_64__
    {
        memory_function::MEMSET_STREAM_SSE2,
        memory_function_type::MEMSET,
        "memset_stream_sse2",
        bool(__builtin_cpu_supports("sse2")),
        { .memset_impl = &memset_stream_sse2 },
    },
    {
        memory_function::READ,
        memory_function_type::READ,
        "read",
        true,
        { .read_impl = &memory_read },
    },
#endif // __x86_64__
#ifdef __ARM_FEATURE_SVE
    {
        memory_function::MEMSET_STREAM_SVE,
        memory_function_type::MEMSET,
        "memset_stream_sve",
        sve_supported(),
        { .memset_impl = &memset_stream_sve },
    },
#endif // __ARM_FEATURE_SVE
};

template<typename T, typename V>
static const auto &enum_lookup(T first, T last, V value)
{
    for (T it = first; it != last; ++it)
        if (it->value == value)
            return *it;
    abort();
}

static string memory_type_name(memory_type type)
{
    return enum_lookup(begin(memory_types), end(memory_types), type).name;
}

static string memory_function_name(memory_function func)
{
    return enum_lookup(begin(memory_functions), end(memory_functions), func).name;
}

static void post(sem_t &sem)
{
    int result = sem_post(&sem);
    assert(result == 0);
}

static void wait(sem_t &sem)
{
    int result = sem_wait(&sem);
    assert(result == 0);
}

struct thread_data
{
    sem_t start_sem;
    sem_t done_sem;
    std::future<void> future;
    bool shutdown = false;

    thread_data()
    {
        int result;
        result = sem_init(&start_sem, 0, 0);
        assert(result == 0);
        result = sem_init(&done_sem, 0, 0);
        assert(result == 0);
    }
};

static void run_passes(
    int passes,
    memory_function mem_func,
    void * __restrict__ dest,
    const void * __restrict__ src,
    size_t buffer_size,
    size_t chunk_size = 0  // 0 means use buffer_size
)
{
    const auto &info = enum_lookup(begin(memory_functions), end(memory_functions), mem_func);
    if (chunk_size == 0)
        chunk_size = buffer_size;
    switch (info.type)
    {
    case memory_function_type::MEMCPY:
        for (int p = 0; p < passes; p++)
        {
            size_t offset = 0;
            while (offset < buffer_size)
            {
                size_t n = min(chunk_size, buffer_size);
                info.impl.memcpy_impl(
                    (void *) ((byte *) dest + offset),
                    (const void *) ((const byte *) src + offset),
                    n
                );
                offset += n;
            }
        }
        break;
    case memory_function_type::MEMSET:
        for (int p = 0; p < passes; p++)
        {
            size_t offset = 0;
            while (offset < buffer_size)
            {
                size_t n = min(chunk_size, buffer_size - offset);
                info.impl.memset_impl((void *) ((byte *) dest + offset), 0, n);
                offset += n;
            }
        }
        break;
    case memory_function_type::READ:
        for (int p = 0; p < passes; p++)
        {
            size_t offset = 0;
            while (offset < buffer_size)
            {
                size_t n = min(chunk_size, buffer_size - offset);
                info.impl.read_impl((const void *) ((const byte *) src + offset), n);
                offset += n;
            }
        }
    }
}

static void self_test()
{
    const size_t buffer_size = 12345;
    const int tail = 64;  // elements to not copy at end
    const byte dummy{123};  // value to write in guard regions
    mt19937 engine;
    uniform_int_distribution<int> distribution(0, 255);
    // Test the copy functions
    vector<byte> dest(buffer_size);
    vector<byte> src(buffer_size);
    vector<byte> backup;
    for (size_t i = 0; i < buffer_size; i++)
        src[i] = byte(distribution(engine));
    backup = src;

    for (const auto &m : memory_functions)
    {
        cout << "Testing " << m.name << " ... " << flush;
        if (!m.supported)
        {
            cout << "skipped (no HW support)\n";
            continue;
        }
        for (int head = 0; head <= 64; head++)
        {
            size_t n = buffer_size - head - tail;
            fill(dest.begin(), dest.end(), dummy);
            run_passes(1, m.value, dest.data() + head, src.data() + head, n);
            // Check that the source didn't get modified
            assert(equal(src.begin(), src.end(), backup.begin()));
            // Check that the guard areas were not touched
            assert(count(dest.begin(), dest.begin() + head, dummy) == head);
            assert(count(dest.end() - tail, dest.end(), dummy) == tail);
            switch (m.type)
            {
            case memory_function_type::MEMCPY:
                // Check that the destination was written correctly
                assert(equal(src.begin() + head, src.end() - tail, dest.begin() + head));
                break;
            case memory_function_type::MEMSET:
                // Check that the destination was cleared
                assert(size_t(count(dest.begin() + head, dest.end() - tail, byte(0))) == n);
                break;
            case memory_function_type::READ:
                // Not much one can test here
                break;
            }
        }
        cout << "ok\n" << flush;
    }
}

static void worker(
    int core,
    size_t buffer_size,
    size_t chunk_size,
    memory_type mem_type,
    memory_function mem_func,
    int src_align,
    int dst_align,
    int passes,
    thread_data &data
)
{
    if (core >= 0)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);
        int result = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        assert(result == 0);
    }
    char *src = allocate(buffer_size + src_align, mem_type) + src_align;
    char *dst = allocate(buffer_size + dst_align, mem_type) + dst_align;
    post(data.done_sem);  // Tell the main thread we're ready for work
    while (true)
    {
        wait(data.start_sem);
        if (data.shutdown)
            break;
        run_passes(passes, mem_func, dst, src, buffer_size, chunk_size);
        post(data.done_sem);
    }
}

template<typename T>
auto parse_enum(T first, T last, const string &value, const string &description)
{
    for (T it = first; it != last; ++it)
        if (it->name == value)
            return it->value;
    cerr << "Invalid " << description << " (must be ";
    for (T it = first; it != last; ++it)
    {
        if (it != first)
            cerr << " / ";
        cerr << it->name;
    }
    cerr << ")\n";
    exit(1);
}

int main(int argc, char *const argv[])
{
    memory_type mem_type = memory_type::MMAP;
    memory_function mem_func = memory_function::MEMCPY;
    size_t buffer_size = 128 * 1024 * 1024;
    size_t chunk_size = 0;
    int src_align = 0, dst_align = 0;  // relative to cache line size
    vector<int> cores;
    long long passes = 10;
    long long repeats = -1;
    bool do_self_test = false;
    int opt;
    while ((opt = getopt(argc, argv, "t:f:b:c:p:r:S:D:T")) != -1)
    {
        switch (opt)
        {
        case 't':
            mem_type = parse_enum(begin(memory_types), end(memory_types), optarg, "memory type");
            break;
        case 'f':
            mem_func = parse_enum(begin(memory_functions), end(memory_functions), optarg, "memory function");
            break;
        case 'b':
            buffer_size = atoll(optarg);
            break;
        case 'c':
            chunk_size = atoll(optarg);
            break;
        case 'p':
            passes = atoll(optarg);
            break;
        case 'r':
            repeats = atoll(optarg);
            break;
        case 'S':
            src_align = atoi(optarg);
            break;
        case 'D':
            dst_align = atoi(optarg);
            break;
        case 'T':
            do_self_test = true;
            break;
        default:
            return 1;
        }
    }
    for (int i = optind; i < argc; i++)
        cores.push_back(atoi(argv[i]));
    if (cores.empty())
        cores.push_back(-1);

    if (do_self_test)
    {
        self_test();
        return 0;
    }
    if (!enum_lookup(begin(memory_functions), end(memory_functions), mem_func).supported)
    {
        cerr << "Memory function " << memory_function_name(mem_func) << " is not supported on this CPU\n";
        return 1;
    }

    cout << "Using " << cores.size() << " threads, each with " << buffer_size << " bytes of "
        << memory_type_name(mem_type) << " memory (" << passes << " passes)\n";
    cout << "Using function " << memory_function_name(mem_func) << '\n';

    size_t n = cores.size();
    vector<thread_data> data(n);
    for (size_t i = 0; i < n; i++)
        data[i].future = async(
            std::launch::async, worker, cores[i],
            buffer_size, chunk_size, mem_type, mem_func, src_align, dst_align,
            passes, ref(data[i])
        );

    // Wait for all threads to signal they've finished the allocation
    for (size_t i = 0; i < n; i++)
        wait(data[i].done_sem);

    auto start = high_resolution_clock::now();
    while (repeats != 0)
    {
        for (size_t i = 0; i < n; i++)
            post(data[i].start_sem);
        for (size_t i = 0; i < n; i++)
            wait(data[i].done_sem);
        auto now = high_resolution_clock::now();
        duration<double> elapsed = now - start;
        double rate = passes / elapsed.count() * n * buffer_size;
        cout << rate / 1e9 << " GB/s" << endl;
        start = now;
        if (repeats > 0)
            repeats--;
    }
    for (size_t i = 0; i < n; i++)
    {
        data[i].shutdown = true;
        post(data[i].start_sem);
        data[i].future.get();
    }
}
