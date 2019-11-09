#include <iostream>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <chrono>

#include <sys/mman.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

static constexpr std::size_t buffer_size = 128 * 1024 * 1024;
static constexpr int passes = 10;

#define HUGE_PAGES 0

static char *allocate(std::size_t size)
{
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (HUGE_PAGES)
        flags |= MAP_HUGETLB;
    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    assert(addr != MAP_FAILED);
    return (char *) addr;
}

int main()
{
    int reps = omp_get_max_threads();
    std::cout << "Using " << reps << " threads\n";
    std::cout << "Using " << (HUGE_PAGES ? "huge" : "normal") << " pages\n";
    vector<char *> src, dst;
    for (int i = 0; i < reps; i++)
    {
        src.push_back(allocate(buffer_size));
        dst.push_back(allocate(buffer_size));
        memset(src.back(), 1, buffer_size);
        memset(dst.back(), 1, buffer_size);
    }
    auto start = high_resolution_clock::now();
    while (true)
    {
        for (int pass = 0; pass < passes; pass++)
        {
#pragma omp parallel for
            for (int i = 0; i < reps; i++)
                std::memcpy(dst[i], src[i], buffer_size);
        }
        auto now = high_resolution_clock::now();
        duration<double> elapsed = now - start;
        double rate = passes / elapsed.count() * reps * buffer_size;
        cout << rate / 1e9 << " GB/s\n";
        start = now;
    }
}
