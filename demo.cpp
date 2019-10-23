#include <iostream>
#include <memory>
#include <utility>
#include <boost/asio.hpp>
#include <spead2/recv_udp_pcap.h>
#include "src/receiver.h"

static constexpr int PACKET_SAMPLES = 4096;
static constexpr int CHUNK_SAMPLES = 1 << 20;

void ready_callback(receiver &recv, std::unique_ptr<in_chunk> &&chunk)
{
    std::cout << "Received chunk with timestamp " << chunk->timestamp << '\n';
    recv.add_chunk(std::move(chunk));
}

std::unique_ptr<in_chunk> alloc_chunk(std::vector<std::vector<std::uint8_t>> &storage)
{
    auto chunk = std::make_unique<in_chunk>();
    for (int pol = 0; pol < 2; pol++)
    {
        std::vector<std::uint8_t> data(CHUNK_SAMPLES * SAMPLE_BITS / 8);
        chunk->storage[pol] = boost::asio::mutable_buffer(data.data(), data.size());
        storage.push_back(std::move(data));
        chunk->present[pol].resize(CHUNK_SAMPLES / PACKET_SAMPLES);
    }
    return chunk;
}

int main()
{
    std::vector<std::vector<std::uint8_t>> storage;
    boost::asio::io_service io_service;

    receiver recv(io_service, PACKET_SAMPLES, CHUNK_SAMPLES, ready_callback);
    for (int i = 0; i < 4; i++)
        recv.add_chunk(alloc_chunk(storage));
    recv.get_stream(0).emplace_reader<spead2::recv::udp_pcap_file_reader>(
        "/mnt/data/bmerry/pcap/dig10s.pcap");

    io_service.run();
}
