#include <iostream>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <chrono>
#include <future>

#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>
#include <emmintrin.h>

using namespace std;
using namespace std::chrono;
using namespace std::literals::string_literals;

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
    MEMSET,
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
    case memory_function::MEMSET: return "memset";
    case memory_function::READ:   return "read";
    default: abort();
    }
}

static char *allocate(std::size_t size, memory_type type)
{
    void *addr;
    if (type == memory_type::MALLOC)
    {
        addr = malloc(size);
        if (addr == nullptr)
            throw std::bad_alloc();
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

/* Read all the data in [src, src + bytes) and do nothing with it. */
static void memory_read(const void *src, std::size_t bytes)
{
    std::uint8_t result1 = 0;
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
    volatile std::uint8_t sink1 = result1;
    volatile __m128i sink2 = result2;
    // Suppress unused variable warnings
    (void) sink1;
    (void) sink2;
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

static void worker(int core, std::size_t buffer_size, memory_type mem_type, memory_function mem_func, int passes, thread_data &data)
{
    if (core >= 0)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);
        int result = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        assert(result == 0);
    }
    char *src = allocate(buffer_size, mem_type);
    char *dst = allocate(buffer_size, mem_type);
    while (true)
    {
        int result = sem_wait(&data.start_sem);
        assert(result == 0);
        switch (mem_func)
        {
        case memory_function::MEMCPY:
            for (int p = 0; p < passes; p++)
                memcpy(dst, src, buffer_size);
            break;
        case memory_function::MEMSET:
            for (int p = 0; p < passes; p++)
                memset(dst, 0, buffer_size);
            break;
        case memory_function::READ:
            for (int p = 0; p < passes; p++)
                memory_read(src, buffer_size);
        }
        result = sem_post(&data.done_sem);
        assert(result == 0);
    }
}

int main(int argc, char *const argv[])
{
    memory_type mem_type = memory_type::MMAP;
    memory_function mem_func = memory_function::MEMCPY;
    std::size_t buffer_size = 128 * 1024 * 1024;
    std::vector<int> cores;
    int passes = 10;
    int opt;
    while ((opt = getopt(argc, argv, "t:f:b:p:")) != -1)
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
            else if (optarg == "memset"s)
                mem_func = memory_function::MEMSET;
            else if (optarg == "read"s)
                mem_func = memory_function::READ;
            else
            {
                std::cerr << "Invalid memory function (must be memcpy, memset or read)\n";
                return 1;
            }
            break;
        case 'b':
            buffer_size = atoll(optarg);
            break;
        case 'p':
            passes = atoi(optarg);
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
            std::launch::async, worker, cores[i], buffer_size, mem_type, mem_func, passes, std::ref(data[i]));
    auto start = high_resolution_clock::now();
    while (true)
    {
        for (size_t i = 0; i < n; i++)
        {
            int result = sem_post(&data[i].start_sem);
            assert(result == 0);
        }
        for (size_t i = 0; i < n; i++)
        {
            int result = sem_wait(&data[i].done_sem);
            assert(result == 0);
        }
        auto now = high_resolution_clock::now();
        duration<double> elapsed = now - start;
        double rate = passes / elapsed.count() * n * buffer_size;
        cout << rate / 1e9 << " GB/s" << endl;
        start = now;
    }
}
