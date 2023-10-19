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
 * -S: an offset to add to the source address
 * -D: an offset to add to the destination address
 * -T: run tests of the function implementations
 *
 * The supported functions are:
 *
 * - memcpy: the library memcpy implementation
 * - memcpy_stream_sse2/memory_stream_avx/memory_stream_avx512: SIMD copies,
 *     using streaming stores
 * - memcpy_rep_movsb: use the x86 "REP MOVSB" instruction
 * - memset: use library memset to clear the destination
 * - memset: use SSE2 streaming stores to clear the destination
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
#include <emmintrin.h>
#include <immintrin.h>

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
    MEMCPY_STREAM_SSE2,
    MEMCPY_STREAM_AVX,
    MEMCPY_STREAM_AVX512,
    MEMCPY_REP_MOVSB,
    MEMSET,
    MEMSET_STREAM_SSE2,
    READ
};

enum class memory_function_type
{
    MEMCPY,
    MEMSET,
    READ
};

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
} memory_functions[] = {
    { memory_function::MEMCPY, memory_function_type::MEMCPY, "memcpy", true },
    { memory_function::MEMCPY_STREAM_SSE2, memory_function_type::MEMCPY, "memcpy_stream_sse2", bool(__builtin_cpu_supports("sse2")) },
    { memory_function::MEMCPY_STREAM_AVX, memory_function_type::MEMCPY, "memcpy_stream_avx", bool(__builtin_cpu_supports("avx")) },
    { memory_function::MEMCPY_STREAM_AVX512, memory_function_type::MEMCPY, "memcpy_stream_avx512", bool(__builtin_cpu_supports("avx512f")) },
    { memory_function::MEMCPY_REP_MOVSB, memory_function_type::MEMCPY, "memcpy_rep_movsb", true },
    { memory_function::MEMSET, memory_function_type::MEMSET, "memset", true },
    { memory_function::MEMSET_STREAM_SSE2, memory_function_type::MEMSET, "memset_stream_sse2", bool(__builtin_cpu_supports("sse2")) },
    { memory_function::READ, memory_function_type::READ, "read", true },
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

/* Wrap another memcpy operation and use it to handle the aligned part of the copy,
 * fixing up the head and tail with std::memcpy. Specifically, the inner
 * function may assume that
 * - dest is aligned to a multiple of A
 * - n is a multiple of M
 */
template<size_t A, size_t M, typename F>
static void *memcpy_align(void * __restrict__ dest, const void * __restrict__ src, size_t n, const F &inner) noexcept
{
    static_assert(A > 0 && !(A & (A - 1)), "A must be a power of 2");
    static_assert(M > 0 && !(M & (M - 1)), "M must be a power of 2");

    void *aligned_dest = dest;
    const void *aligned_src = src;  // not necessarily aligned, but corresponds to aligned_dest
    size_t aligned_n = n;
    if (!align(A, M, aligned_dest, aligned_n))
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
    // Round aligned_n down to a multiple of M
    size_t truncated_n = aligned_n & ~(M - 1);

    inner(aligned_dest, aligned_src, truncated_n);

    size_t tail = aligned_n - truncated_n;
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
static void *memcpy_stream_sse2_impl(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    for (size_t offset = 0; offset + 64 <= n; offset += 64)
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

static void *memcpy_stream_sse2(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_align<cache_line_size, 64>(dest, src, n, memcpy_stream_sse2_impl);
}

// memcpy, with AVX streaming stores (requires dest to be 32-byte aligned and n to be a multiple of 64)
[[gnu::target("avx2")]]
static void *memcpy_stream_avx_impl(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    size_t offset = 0;
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

static void *memcpy_stream_avx(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_align<cache_line_size, 64>(dest, src, n, memcpy_stream_avx_impl);
}

// memcpy, with AVX-512 streaming stores (requires 64-byte alignment)
[[gnu::target("avx512f")]]
static void *memcpy_stream_avx512_impl(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    size_t offset = 0;
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

static void *memcpy_stream_avx512(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    return memcpy_align<64, 64>(dest, src, n, memcpy_stream_avx512_impl);
}

static void *memcpy_rep_movsb(void * __restrict__ dest, const void * __restrict__ src, size_t n) noexcept
{
    void *orig_dest = dest;
    asm volatile("rep movsb" : "+c" (n), "+D" (dest), "+S" (src) : : "memory");
    return orig_dest;
}

/* memset, but using SSE streaming stores */
static void memset_stream_sse2(void *dst, int c, size_t bytes) noexcept
{
    // Simplifies some edge cases
    if (bytes <= 16)
    {
        std::memset(dst, c, bytes);
        return;
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
    size_t buffer_size
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
        run_passes(passes, mem_func, dst, src, buffer_size);
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
    int src_align = 0, dst_align = 0;  // relative to cache line size
    vector<int> cores;
    long long passes = 10;
    long long repeats = -1;
    bool do_self_test = false;
    int opt;
    while ((opt = getopt(argc, argv, "t:f:b:p:r:S:D:T")) != -1)
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
            buffer_size, mem_type, mem_func, src_align, dst_align,
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
