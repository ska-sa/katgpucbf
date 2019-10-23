#include <iostream>
#include <memory>
#include <utility>
#include <numeric>
#include <boost/asio.hpp>
#include <spead2/recv_udp_pcap.h>
#include "src/receiver.h"

static constexpr int PACKET_SAMPLES = 4096;
static constexpr int CHUNK_SAMPLES = 1 << 25;

class plain_in_chunk : public in_chunk
{
private:
    std::unique_ptr<std::uint8_t[]> storage_ptr[N_POL];

public:
    plain_in_chunk()
    {
        constexpr std::size_t bytes = CHUNK_SAMPLES * SAMPLE_BITS / 8;
        for (int pol = 0; pol < 2; pol++)
        {
            storage_ptr[pol] = std::make_unique<std::uint8_t[]>(bytes);
            storage[pol] = boost::asio::mutable_buffer(storage_ptr[pol].get(), bytes);
            present[pol].resize(CHUNK_SAMPLES / PACKET_SAMPLES);
        }
    }
};

int main()
{
    std::vector<std::vector<std::uint8_t>> storage;

    receiver recv(PACKET_SAMPLES, CHUNK_SAMPLES);
    for (int i = 0; i < 4; i++)
        recv.add_chunk(std::make_unique<plain_in_chunk>());
    recv.get_stream(0).emplace_reader<spead2::recv::udp_pcap_file_reader>(
        "/mnt/data/bmerry/pcap/dig1s.pcap");
    recv.get_stream(1).emplace_reader<spead2::recv::udp_pcap_file_reader>(
        "/mnt/data/bmerry/pcap/dig1s.pcap");

    while (true)
    {
        try
        {
            auto chunk = recv.ringbuffer.pop();
            std::size_t good = 0;
            for (int i = 0; i < N_POL; i++)
                good = std::accumulate(chunk->present[i].begin(), chunk->present[i].end(), good);
            std::cout << "Received chunk with timestamp " << chunk->timestamp
                << " (" << good << " / " << N_POL * chunk->present[0].size() << " packets)\n";
            recv.add_chunk(std::move(chunk));
        }
        catch (spead2::ringbuffer_stopped &e)
        {
            break;
        }
    }
}
