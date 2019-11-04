#include <iostream>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <chrono>

#include <sys/mman.h>

using namespace std;
using namespace std::chrono;

static constexpr std::size_t buffer_size = 128 * 1024 * 1024;
static constexpr int reps = 10;
static constexpr int passes = 5;

static char *allocate(std::size_t size)
{
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (0)
    {
        flags |= MAP_HUGETLB;
        std::cout << "Using huge pages\n";
    }
    else
    {
        std::cout << "Using normal pages\n";
    }
    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    assert(addr != MAP_FAILED);
    return (char *) addr;
}

int main()
{
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
