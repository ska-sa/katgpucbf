#include <iostream>
#include <memory>
#include <utility>
#include <numeric>
#include <boost/asio.hpp>
#include "src/recv.h"

using namespace katfgpu::recv;

static constexpr int SAMPLE_BITS = 10;
static constexpr std::size_t PACKET_SAMPLES = 4096;
static constexpr std::size_t CHUNK_SAMPLES = 1 << 28;

class plain_chunk : public chunk
{
private:
    std::unique_ptr<std::uint8_t[]> storage_ptr;

public:
    plain_chunk()
    {
        constexpr std::size_t bytes = CHUNK_SAMPLES / 8 * SAMPLE_BITS;
        storage_ptr = std::make_unique<std::uint8_t[]>(bytes);
        storage = boost::asio::mutable_buffer(storage_ptr.get(), bytes);
        present.resize(CHUNK_SAMPLES / PACKET_SAMPLES);
    }
};

int main()
{
    static constexpr int N_POL = 2;
    stream::ringbuffer_t ringbuffer(2);
    std::array<std::unique_ptr<stream>, N_POL> recv;
    for (int pol = 0; pol < N_POL; pol++)
    {
        recv[pol] = std::make_unique<stream>(pol, SAMPLE_BITS, PACKET_SAMPLES, CHUNK_SAMPLES, ringbuffer);
        for (int i = 0; i < 4; i++)
            recv[pol]->add_chunk(std::make_unique<plain_chunk>());
    }
    for (int pol = 0; pol < N_POL; pol++)
    {
#if 1
        recv[pol]->add_udp_pcap_file_reader("/mnt/data/bmerry/pcap/dig1s.pcap");
#else
        std::vector<std::pair<std::string, std::uint16_t>> endpoints;
        for (int i = 0; i < 8; i++)
            endpoints.emplace_back("239.10.0." + std::to_string(pol * 8 + i), 7148);
        recv[pol]->add_udp_ibv_reader(endpoints, "10.100.24.69", 32 * 1024 * 1024, pol);
#endif
    }

    while (true)
    {
        try
        {
            auto c = ringbuffer.pop();
            std::size_t good = 0;
            good = std::accumulate(c->present.begin(), c->present.end(), good);
            std::cout << "Received chunk with timestamp " << c->timestamp
                << " on pol " << c->pol
                << " (" << good << " / " << c->present.size() << " packets)\n";
            recv[c->pol]->add_chunk(std::move(c));
        }
        catch (spead2::ringbuffer_stopped &e)
        {
            break;
        }
    }
    for (int pol = 0; pol < N_POL; pol++)
        recv[pol]->stop();
}
