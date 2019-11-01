#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>
#include <boost/program_options.hpp>
#include <spead2/send_udp_ibv.h>
#include <spead2/common_semaphore.h>

namespace po = boost::program_options;

static constexpr double bandwidth = 1712000000.0;
static constexpr int capacity = 65536;
static constexpr int sample_bits = 10;
static constexpr int packet_samples = 4096;
static constexpr std::size_t packet_size = packet_samples * sample_bits / 8;
static const spead2::flavour flavour(4, 64, 48);
static const spead2::send::stream_config config(
    8872,      // Doesn't matter, just needs to be bigger than actual size
    bandwidth * 10.0 / 8.0 * (packet_size + 64) / packet_size, 65536, 128);

struct heap_data
{
    std::unique_ptr<std::uint8_t[]> data;
    spead2::send::heap heap;
    spead2::send::heap::item_handle timestamp_handle;

    heap_data()
        : data(std::make_unique<std::uint8_t[]>(packet_size)),
        heap(flavour),
        timestamp_handle(heap.add_item(0x1600, 0))
    {
        heap.add_item(0x3101, 0);
        heap.add_item(0x3102, 0);
        heap.add_item(0x3300, data.get(), packet_size, false);
        heap.set_repeat_pointers(true);
    }
};

struct polarisation
{
    spead2::thread_pool pool;
    spead2::send::udp_ibv_stream stream;
    std::vector<heap_data> heaps;
    spead2::semaphore next_sem;
    int next = 0;
    std::int64_t timestamp = 0;

    polarisation(std::size_t capacity,
                 const boost::asio::ip::udp::endpoint &endpoint,
                 const boost::asio::ip::address &interface_address)
        : pool(1),
        stream(pool, endpoint, config, interface_address),
        next_sem(config.get_max_heaps())
    {
        heaps.reserve(capacity);
        for (std::size_t i = 0; i < capacity; i++)
            heaps.emplace_back();
    }

    void send_next()
    {
        spead2::semaphore &next_sem = this->next_sem;
        auto callback = [&next_sem] (const boost::system::error_code &ec, std::size_t)
        {
            if (ec)
                std::cerr << "Error: " << ec;
            else
                next_sem.put();
        };
        heaps[next].heap.get_item(heaps[next].timestamp_handle).data.immediate = timestamp;
        timestamp += packet_samples;
        stream.async_send_heap(heaps[next].heap, callback);
        next = (next + 1) % heaps.size();
    }
};

int main(int argc, char **argv)
{
    boost::asio::ip::udp::endpoint endpoint0(
        boost::asio::ip::address::from_string("239.101.200.0"), 7148);
    boost::asio::ip::udp::endpoint endpoint1(
        boost::asio::ip::address::from_string("239.101.200.2"), 7148);
    auto interface_address = boost::asio::ip::address::from_string("192.168.8.2");

    std::array<std::unique_ptr<polarisation>, 2> pols;
    pols[0] = std::make_unique<polarisation>(capacity, endpoint0, interface_address);
    pols[1] = std::make_unique<polarisation>(capacity, endpoint1, interface_address);
    while (true)
    {
        for (int p = 0; p < 2; p++)
            semaphore_get(pols[p]->next_sem);
        for (int p = 0; p < 2; p++)
            pols[p]->send_next();
    }
    return 0;
}
