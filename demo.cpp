#include <iostream>
#include <memory>
#include <utility>
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

void ready_callback(receiver &recv, std::unique_ptr<in_chunk> &&chunk)
{
    std::cout << "Received chunk with timestamp " << chunk->timestamp << '\n';
    recv.add_chunk(std::move(chunk));
}

int main()
{
    std::vector<std::vector<std::uint8_t>> storage;
    boost::asio::io_service io_service;

    receiver recv(io_service, PACKET_SAMPLES, CHUNK_SAMPLES, ready_callback);
    for (int i = 0; i < 4; i++)
        recv.add_chunk(std::make_unique<plain_in_chunk>());
    recv.get_stream(0).emplace_reader<spead2::recv::udp_pcap_file_reader>(
        "/mnt/data/bmerry/pcap/dig10s.pcap");

    io_service.run();
}
