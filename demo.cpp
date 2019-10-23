#include <iostream>
#include <memory>
#include <utility>
#include <numeric>
#include <boost/asio.hpp>
#include <spead2/recv_udp_pcap.h>
#include "src/receiver.h"

static constexpr std::size_t PACKET_SAMPLES = 4096;
static constexpr std::size_t CHUNK_SAMPLES = 1 << 28;

class plain_in_chunk : public in_chunk
{
private:
    std::unique_ptr<std::uint8_t[]> storage_ptr;

public:
    plain_in_chunk()
    {
        constexpr std::size_t bytes = CHUNK_SAMPLES * SAMPLE_BITS / 8;
        storage_ptr = std::make_unique<std::uint8_t[]>(bytes);
        storage = boost::asio::mutable_buffer(storage_ptr.get(), bytes);
        present.resize(CHUNK_SAMPLES / PACKET_SAMPLES);
    }
};

int main()
{
    static constexpr int N_POL = 2;
    receiver::ringbuffer_t ringbuffer(2);
    std::array<std::unique_ptr<receiver>, N_POL> recv;
    for (int pol = 0; pol < N_POL; pol++)
    {
        recv[pol] = std::make_unique<receiver>(pol, PACKET_SAMPLES, CHUNK_SAMPLES, ringbuffer);
        for (int i = 0; i < 4; i++)
            recv[pol]->add_chunk(std::make_unique<plain_in_chunk>());
    }
    for (int pol = 0; pol < N_POL; pol++)
    {
        recv[pol]->get_stream().emplace_reader<spead2::recv::udp_pcap_file_reader>(
            "/mnt/data/bmerry/pcap/dig1s.pcap");
    }

    while (true)
    {
        try
        {
            auto chunk = ringbuffer.pop();
            std::size_t good = 0;
            good = std::accumulate(chunk->present.begin(), chunk->present.end(), good);
            std::cout << "Received chunk with timestamp " << chunk->timestamp
                << " on pol " << chunk->pol
                << " (" << good << " / " << chunk->present.size() << " packets)\n";
            recv[chunk->pol]->add_chunk(std::move(chunk));
        }
        catch (spead2::ringbuffer_stopped &e)
        {
            break;
        }
    }
    for (int pol = 0; pol < N_POL; pol++)
        recv[pol]->get_stream().stop();   // TODO: make a stop function in recv
}
