/*******************************************************************************
 * Copyright (c) 2019-2020, 2022-2023, National Research Foundation (SARAO)
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
#include <emmintrin.h>
#include <immintrin.h>

using namespace std;
using namespace std::chrono;
using namespace std::literals::string_literals;

static constexpr std::size_t cache_line_size = 64;  // guesstimate

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
    MEMCPY_STREAM_SSE2,
    MEMCPY_STREAM_AVX,
    MEMCPY_STREAM_AVX512,
    MEMCPY_REP_MOVSB,
    MEMSET,
    MEMSET_STREAM_SSE2,
    READ
};

static string memory_type_name(memory_type type)
{
    switch (type)
    {
    case memory_type::MALLOC:    return "malloc";
    case memory_type::MMAP:      return "mmap";
    case memory_type::MMAP_HUGE: return "mmap_huge";
    case memory_type::MADV_HUGE: return "madv_huge";
    default: abort();
    }
}

static string memory_function_name(memory_function func)
{
    switch (func)
    {
    case memory_function::MEMCPY: return "memcpy";
    case memory_function::MEMCPY_STREAM_SSE2: return "memcpy_stream_sse2";
    case memory_function::MEMCPY_STREAM_AVX: return "memcpy_stream_avx";
    case memory_function::MEMCPY_STREAM_AVX512: return "memcpy_stream_avx512";
    case memory_function::MEMCPY_REP_MOVSB: return "memcpy_rep_movsb";
    case memory_function::MEMSET: return "memset";
    case memory_function::MEMSET_STREAM_SSE2: return "memset_stream_sse2";
    case memory_function::READ:   return "read";
    default: abort();
    }
}

static char *allocate(std::size_t size, memory_type type)
{
    void *addr;
    if (type == memory_type::MALLOC)
    {
        // Ensure at least cache line alignment
        std::size_t space = size + cache_line_size;
        addr = malloc(size + cache_line_size);
        if (addr == nullptr)
            throw std::bad_alloc();
        std::align(cache_line_size, size, addr, space);
    }
    else
    {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (type == memory_type::MMAP_HUGE)
            flags |= MAP_HUGETLB;
        addr = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (addr == MAP_FAILED)
            throw std::bad_alloc();
        if (type == memory_type::MADV_HUGE)
            madvise(addr, size, MADV_HUGEPAGE);
    }
    memset(addr, 1, size);  // ensure it has real pages
    return (char *) addr;
}

/* Wrap another memcpy operation and use it to handle the aligned part of the copy,
 * fixing up the head and tail with std::memcpy. Specifically, the inner
 * function may assume that
 * - dest is aligned to a multiple of A
 * - n is a multiple of M
 */
template<std::size_t A, std::size_t M, typename F>
static void *memcpy_align(void * __restrict__ dest, const void * __restrict__ src, std::size_t n, const F &inner) noexcept
{
    static_assert(A > 0 && !(A & (A - 1)), "A must be a power of 2");
    static_assert(M > 0 && !(M & (M - 1)), "M must be a power of 2");

    void *aligned_dest = dest;
    const void *aligned_src = src;  // not necessarily aligned, but corresponds to aligned_dest
    std::size_t aligned_n = n;
    if (!std::align(A, M, aligned_dest, aligned_n))
    {
        // Not even room for one aligned block. Just fall back to plain memcpy
        return std::memcpy(dest, src, n);
    }
    // Copy the head
    std::size_t head = (char *) aligned_dest - (char *) dest;
    if (head != 0)
    {
        std::memcpy(dest, src, head);
        aligned_src = (const void *) ((const char *) src + head);
    }
    // Round aligned_n down to a multiple of M
    std::size_t truncated_n = aligned_n & ~(M - 1);

    inner(aligned_dest, aligned_src, truncated_n);

    std::size_t tail = aligned_n - truncated_n;
    if (tail != 0)
    {
        std::memcpy(
            (void *) ((char *) aligned_dest + truncated_n),
            (const void *) ((const char *) aligned_src + truncated_n),
            tail
        );
    }

    return dest;
}

// memcpy, with SSE2 streaming stores (requires dest to be 16-byte aligned and n to be a multiple of 64)
static void *memcpy_stream_sse2_impl(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    for (std::size_t offset = 0; offset + 64 <= n; offset += 64)
    {
        __m128i value0 = _mm_loadu_si128((__m128i const *) (src_c + offset + 0));
        __m128i value1 = _mm_loadu_si128((__m128i const *) (src_c + offset + 16));
        __m128i value2 = _mm_loadu_si128((__m128i const *) (src_c + offset + 32));
        __m128i value3 = _mm_loadu_si128((__m128i const *) (src_c + offset + 48));
        _mm_stream_si128((__m128i *) (dest_c + offset + 0), value0);
        _mm_stream_si128((__m128i *) (dest_c + offset + 16), value1);
        _mm_stream_si128((__m128i *) (dest_c + offset + 32), value2);
        _mm_stream_si128((__m128i *) (dest_c + offset + 48), value3);
    }
    _mm_sfence();
    return dest;
}

static void *memcpy_stream_sse2(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    return memcpy_align<cache_line_size, 64>(dest, src, n, memcpy_stream_sse2_impl);
}

// memcpy, with AVX streaming stores (requires dest to be 32-byte aligned and n to be a multiple of 64)
[[gnu::target("avx2")]]
static void *memcpy_stream_avx_impl(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    std::size_t offset = 0;
    for (offset = 0; offset + 256 <= n; offset += 256)
    {
        __m256i value0 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32 * 0));
        __m256i value1 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32 * 1));
        __m256i value2 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32 * 2));
        __m256i value3 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32 * 3));
        __m256i value4 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32 * 4));
        __m256i value5 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32 * 5));
        __m256i value6 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32 * 6));
        __m256i value7 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32 * 7));
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32 * 0), value0);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32 * 1), value1);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32 * 2), value2);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32 * 3), value3);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32 * 4), value4);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32 * 5), value5);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32 * 6), value6);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32 * 7), value7);
    }
    for (; offset + 64 <= n; offset += 64)
    {
        __m256i value0 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 0));
        __m256i value1 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32));
        _mm256_stream_si256((__m256i *) (dest_c + offset + 0), value0);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32), value1);
    }
    _mm_sfence();
    return dest;
}

static void *memcpy_stream_avx(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    return memcpy_align<cache_line_size, 64>(dest, src, n, memcpy_stream_avx_impl);
}

// memcpy, with AVX-512 streaming stores (requires 64-byte alignment)
[[gnu::target("avx512f")]]
static void *memcpy_stream_avx512_impl(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    std::size_t offset = 0;
    for (offset = 0; offset + 512 <= n; offset += 512)
    {
        __m512i value0 = _mm512_loadu_si512((__m512i const *) (src_c + offset + 64 * 0));
        __m512i value1 = _mm512_loadu_si512((__m512i const *) (src_c + offset + 64 * 1));
        __m512i value2 = _mm512_loadu_si512((__m512i const *) (src_c + offset + 64 * 2));
        __m512i value3 = _mm512_loadu_si512((__m512i const *) (src_c + offset + 64 * 3));
        __m512i value4 = _mm512_loadu_si512((__m512i const *) (src_c + offset + 64 * 4));
        __m512i value5 = _mm512_loadu_si512((__m512i const *) (src_c + offset + 64 * 5));
        __m512i value6 = _mm512_loadu_si512((__m512i const *) (src_c + offset + 64 * 6));
        __m512i value7 = _mm512_loadu_si512((__m512i const *) (src_c + offset + 64 * 7));
        _mm512_stream_si512((__m512i *) (dest_c + offset + 64 * 0), value0);
        _mm512_stream_si512((__m512i *) (dest_c + offset + 64 * 1), value1);
        _mm512_stream_si512((__m512i *) (dest_c + offset + 64 * 2), value2);
        _mm512_stream_si512((__m512i *) (dest_c + offset + 64 * 3), value3);
        _mm512_stream_si512((__m512i *) (dest_c + offset + 64 * 4), value4);
        _mm512_stream_si512((__m512i *) (dest_c + offset + 64 * 5), value5);
        _mm512_stream_si512((__m512i *) (dest_c + offset + 64 * 6), value6);
        _mm512_stream_si512((__m512i *) (dest_c + offset + 64 * 7), value7);
    }
    for (; offset + 64 <= n; offset += 64)
    {
        __m512i value0 = _mm512_loadu_si512((__m512i const *) (src_c + offset + 0));
        _mm512_stream_si512((__m512i *) (dest_c + offset + 0), value0);
    }
    _mm_sfence();
    return dest;
}

static void *memcpy_stream_avx512(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    return memcpy_align<64, 64>(dest, src, n, memcpy_stream_avx512_impl);
}

static void *memcpy_rep_movsb(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    void *orig_dest = dest;
    asm volatile("rep movsb" : "+c" (n), "+D" (dest), "+S" (src) : : "memory");
    return orig_dest;
}

/* memset, but using SSE streaming stores */
static void memset_stream_sse2(void *dst, int c, std::size_t bytes) noexcept
{
    // Simplifies some edge cases
    if (bytes <= 16)
    {
        std::memset(dst, c, bytes);
        return;
    }

    // Process prefix up to 16-byte alignment
    char *cdst = (char *) dst;
    char *cdst_round = (char *) ((std::uintptr_t(dst) + 0xf) & ~0xf);
    if (cdst != cdst_round)
    {
        std::memset(dst, c, cdst_round - cdst);
        bytes -= cdst_round - cdst;
    }

    // Use streaming stores for the bulk
    __m128i value;
    std::memset(&value, 0, sizeof(value));
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
}

/* Read all the data in [src, src + bytes) and do nothing with it. */
static void memory_read(const void *src, std::size_t bytes) noexcept
{
    std::uint8_t result1 = 0;
    // Process prefix up to 16-byte alignment
    const char *csrc = (const char *) src;
    while (((std::uintptr_t) csrc) & 0xf)
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
    volatile std::uint8_t sink1 = result1;
    volatile __m128i sink2 = result2;
    // Suppress unused variable warnings
    (void) sink1;
    (void) sink2;
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
    std::size_t buffer_size
)
{
    switch (mem_func)
    {
    case memory_function::MEMCPY:
        for (int p = 0; p < passes; p++)
            memcpy(dest, src, buffer_size);
        break;
    case memory_function::MEMCPY_STREAM_SSE2:
        for (int p = 0; p < passes; p++)
            memcpy_stream_sse2(dest, src, buffer_size);
        break;
    case memory_function::MEMCPY_STREAM_AVX:
        for (int p = 0; p < passes; p++)
            memcpy_stream_avx(dest, src, buffer_size);
        break;
    case memory_function::MEMCPY_STREAM_AVX512:
        for (int p = 0; p < passes; p++)
            memcpy_stream_avx512(dest, src, buffer_size);
        break;
    case memory_function::MEMCPY_REP_MOVSB:
        for (int p = 0; p < passes; p++)
            memcpy_rep_movsb(dest, src, buffer_size);
        break;
    case memory_function::MEMSET:
        for (int p = 0; p < passes; p++)
            memset(dest, 0, buffer_size);
        break;
    case memory_function::MEMSET_STREAM_SSE2:
        for (int p = 0; p < passes; p++)
            memset_stream_sse2(dest, 0, buffer_size);
        break;
    case memory_function::READ:
        for (int p = 0; p < passes; p++)
            memory_read(src, buffer_size);
    }
}

static void self_test(
    memory_function mem_func,
    void * __restrict__ dest,
    void * __restrict__ src,
    std::size_t buffer_size)
{
    std::mt19937 engine;
    std::uniform_int_distribution<int> distribution(0, 255);
    unsigned char *dest_c = static_cast<unsigned char *>(dest);
    unsigned char *src_c = static_cast<unsigned char *>(src);
    for (std::size_t i = 0; i < buffer_size; i++)
    {
        dest_c[i] = distribution(engine);
        src_c[i] = distribution(engine);
    }
    run_passes(1, mem_func, dest, src, buffer_size);
    switch (mem_func)
    {
    case memory_function::MEMCPY:
    case memory_function::MEMCPY_STREAM_SSE2:
    case memory_function::MEMCPY_STREAM_AVX:
    case memory_function::MEMCPY_STREAM_AVX512:
    case memory_function::MEMCPY_REP_MOVSB:
        // TODO: should also keep a backup copy to ensure that the src
        // wasn't modified
        assert(std::memcmp(dest, src, buffer_size) == 0);
        break;
    case memory_function::MEMSET:
    case memory_function::MEMSET_STREAM_SSE2:
        assert(std::size_t(std::count(dest_c, dest_c + buffer_size, 0)) == buffer_size);
        break;
    case memory_function::READ:
        /* Not much one can test here */
        break;
    }
}

static void worker(
    int core,
    std::size_t buffer_size,
    memory_type mem_type,
    memory_function mem_func,
    int src_align,
    int dst_align,
    int passes,
    bool do_self_test,
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
    if (do_self_test)
        self_test(mem_func, dst, src, buffer_size);
    post(data.done_sem);  // Tell the main thread we're ready for work
    while (true)
    {
        wait(data.start_sem);
        run_passes(passes, mem_func, dst, src, buffer_size);
        post(data.done_sem);
    }
}

int main(int argc, char *const argv[])
{
    memory_type mem_type = memory_type::MMAP;
    memory_function mem_func = memory_function::MEMCPY;
    std::size_t buffer_size = 128 * 1024 * 1024;
    int src_align = 0, dst_align = 0;  // relative to cache line size
    std::vector<int> cores;
    int passes = 10;
    bool do_self_test = false;
    int opt;
    while ((opt = getopt(argc, argv, "t:f:b:p:S:D:T")) != -1)
    {
        switch (opt)
        {
        case 't':
            if (optarg == "malloc"s)
                mem_type = memory_type::MALLOC;
            else if (optarg == "mmap"s)
                mem_type = memory_type::MMAP;
            else if (optarg == "mmap_huge"s)
                mem_type = memory_type::MMAP_HUGE;
            else if (optarg == "madv_huge"s)
                mem_type = memory_type::MADV_HUGE;
            else
            {
                std::cerr << "Invalid memory type (must be malloc, mmap, mmap_huge or madv_huge)\n";
                return 1;
            }
            break;
        case 'f':
            if (optarg == "memcpy"s)
                mem_func = memory_function::MEMCPY;
            else if (optarg == "memcpy_stream_sse2"s)
                mem_func = memory_function::MEMCPY_STREAM_SSE2;
            else if (optarg == "memcpy_stream_avx"s)
                mem_func = memory_function::MEMCPY_STREAM_AVX;
            else if (optarg == "memcpy_stream_avx512"s)
                mem_func = memory_function::MEMCPY_STREAM_AVX512;
            else if (optarg == "memcpy_rep_movsb"s)
                mem_func = memory_function::MEMCPY_REP_MOVSB;
            else if (optarg == "memset"s)
                mem_func = memory_function::MEMSET;
            else if (optarg == "memset_stream_sse2"s)
                mem_func = memory_function::MEMSET_STREAM_SSE2;
            else if (optarg == "read"s)
                mem_func = memory_function::READ;
            else
            {
                std::cerr << "Invalid memory function (must be memcpy, memcpy_stream_sse2, memcpy_stream_avx, memcpy_stream_avx512, memset, memset_stream_sse2 or read)\n";
                return 1;
            }
            break;
        case 'b':
            buffer_size = atoll(optarg);
            break;
        case 'p':
            passes = atoi(optarg);
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

    std::cout << "Using " << cores.size() << " threads, each with " << buffer_size << " bytes of "
        << memory_type_name(mem_type) << " memory (" << passes << " passes)\n";
    std::cout << "Using function " << memory_function_name(mem_func) << '\n';

    size_t n = cores.size();
    std::vector<thread_data> data(n);
    for (size_t i = 0; i < n; i++)
        data[i].future = std::async(
            std::launch::async, worker, cores[i],
            buffer_size, mem_type, mem_func, src_align, dst_align,
            passes, do_self_test, std::ref(data[i])
        );

    // Wait for all threads to signal they've finished the allocation
    for (size_t i = 0; i < n; i++)
        wait(data[i].done_sem);

    auto start = high_resolution_clock::now();
    while (true)
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
    }
}
