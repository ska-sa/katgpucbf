#include <iostream>
#include <vector>
#include <cstddef>
#include <cstring>
#include <chrono>

using namespace std;
using namespace std::chrono;

static constexpr std::size_t buffer_size = 128 * 1024 * 1024;
static constexpr int reps = 20;
static constexpr int passes = 5;

int main()
{
    vector<char *> src, dst;
    for (int i = 0; i < reps; i++)
    {
        src.push_back(new char[buffer_size]);
        dst.push_back(new char[buffer_size]);
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
