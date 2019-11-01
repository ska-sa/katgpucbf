#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <boost/program_options.hpp>
#include <spead2/send_udp_ibv.h>
#include <spead2/common_semaphore.h>

namespace po = boost::program_options;

struct options
{
    std::string interface;
    std::vector<std::string> addresses;
    std::uint16_t port = 7148;
    int max_heaps = 128;
    double adc_rate = 1712000000.0;
};

static constexpr int capacity = 65536;
static constexpr int sample_bits = 10;
static constexpr int packet_samples = 4096;
static constexpr std::size_t packet_size = packet_samples * sample_bits / 8;
static const spead2::flavour flavour(4, 64, 48);

template<typename T>
static po::typed_value<T> *make_opt(T &var)
{
    return po::value<T>(&var)->default_value(var);
}

static options parse_options(int argc, const char **argv)
{
    options opts;
    po::options_description desc, hidden, all;
    desc.add_options()
        ("interface", po::value(&opts.interface)->required(), "Interface address")
        ("port", make_opt(opts.port), "Destination UDP port")
        ("max-heaps", make_opt(opts.max_heaps), "Depth of send queue (per polarisation)")
        ("adc-rate", make_opt(opts.adc_rate), "Sampling rate")
    ;

    hidden.add_options()
        ("address", po::value<std::vector<std::string>>(&opts.addresses)->composing(),
         "destination IP addresses (one per polarisation)")
    ;
    all.add(desc);
    all.add(hidden);
    po::positional_options_description positional;
    positional.add("address", -1);
    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
            .options(all)
            .positional(positional)
            .run(), vm);
        po::notify(vm);
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        std::exit(2);
    }
    return opts;
}

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

    polarisation(const options &opts,
                 std::size_t capacity,
                 const boost::asio::ip::udp::endpoint &endpoint,
                 const boost::asio::ip::address &interface_address)
        : pool(1),
        stream(pool, endpoint,
               spead2::send::stream_config(
                   packet_size + 128,  // Doesn't matter, just needs to be bigger than actual size
                   opts.adc_rate * 10.0 / 8.0 * (packet_size + 72) / packet_size,
                   65536, opts.max_heaps),
               interface_address),
        next_sem(opts.max_heaps)
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

int main(int argc, const char **argv)
{
    options opts = parse_options(argc, argv);

    auto interface_address = boost::asio::ip::address::from_string(opts.interface);
    std::vector<std::unique_ptr<polarisation>> pols;
    for (const std::string &address : opts.addresses)
    {
        boost::asio::ip::udp::endpoint endpoint(
            boost::asio::ip::address::from_string(address), opts.port);
        pols.push_back(std::make_unique<polarisation>(opts, capacity, endpoint, interface_address));
    }
    while (true)
    {
        for (auto &p : pols)
            semaphore_get(p->next_sem);
        for (auto &p : pols)
            p->send_next();
    }
    return 0;
}
