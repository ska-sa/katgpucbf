#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <boost/program_options.hpp>
#include <spead2/send_udp_ibv.h>
#include <spead2/common_semaphore.h>

namespace po = boost::program_options;

struct options
{
    std::string interface;
    std::vector<std::string> addresses;
    int max_heaps = 128;
    double adc_rate = 1712000000.0;
    double signal_freq = 400e6;
};

static constexpr int capacity = 128;
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
        ("max-heaps", make_opt(opts.max_heaps), "Depth of send queue (per polarisation)")
        ("adc-rate", make_opt(opts.adc_rate), "Sampling rate")
    ;

    hidden.add_options()
        ("address", po::value<std::vector<std::string>>(&opts.addresses)->composing(),
         "destination addresses, in form x.x.x.x+N:port (one per polarisation)")
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

static std::vector<boost::asio::ip::udp::endpoint> parse_endpoint_list(const std::string &arg)
{
    std::vector<boost::asio::ip::udp::endpoint> out;

    auto colon = arg.find(':');
    if (colon == std::string::npos)
        throw std::invalid_argument("Address must contain a colon");
    std::uint16_t port = boost::lexical_cast<std::uint16_t>(arg.substr(colon + 1));
    auto plus = arg.find('+');
    if (plus < colon)
    {
        std::string start_str = arg.substr(0, plus);
        auto start = boost::asio::ip::address_v4::from_string(start_str);
        int count = boost::lexical_cast<int>(arg.substr(plus + 1, colon - plus - 1));
        for (int i = 0; i <= count; i++)
        {
            boost::asio::ip::address_v4 addr(start.to_ulong() + i);
            out.emplace_back(addr, port);
        }
    }
    else
    {
        auto addr = boost::asio::ip::address_v4::from_string(arg.substr(0, colon));
        out.emplace_back(addr, port);
    }
    return out;
}

struct heap_data
{
    std::unique_ptr<std::uint8_t[]> data;
    spead2::send::heap heap;
    spead2::send::heap::item_handle timestamp_handle;

    explicit heap_data(const options &opts, std::int64_t timestamp)
        : data(std::make_unique<std::uint8_t[]>(packet_size)),
        heap(flavour),
        timestamp_handle(heap.add_item(0x1600, 0))
    {
        heap.add_item(0x3101, 0);
        heap.add_item(0x3102, 0);
        heap.add_item(0x3300, data.get(), packet_size, false);
        heap.set_repeat_pointers(true);

        float angle_scale = opts.signal_freq / opts.adc_rate * 2 * M_PI;
        unsigned int buffer = 0;
        int buffer_bits = 0;
        int pos = 0;
        for (std::size_t i = 0; i < packet_samples; i++)
        {
            float angle = angle_scale * (timestamp + i);
            int sample = (int) std::round(std::sin(angle) * 256.0);
            buffer = (buffer << 10) | (sample & 0x3ff);
            buffer_bits += 10;
            while (buffer_bits >= 8)
            {
                buffer_bits -= 8;
                data[pos] = buffer >> buffer_bits;
                pos++;
            }
        }
    }
};

struct polarisation
{
    std::vector<boost::asio::ip::udp::endpoint> endpoints;
    std::vector<heap_data> heaps;
    std::size_t next_endpoint = 0;
    std::size_t next = 0;
    std::int64_t timestamp = 0;

    polarisation(const options &opts,
                 std::size_t capacity,
                 const std::vector<boost::asio::ip::udp::endpoint> &endpoints)
        : endpoints(endpoints)
    {
        heaps.reserve(capacity);
        for (std::size_t i = 0; i < capacity; i++)
            heaps.emplace_back(opts, i * packet_samples);
    }

    template<typename Callback>
    void send_next(spead2::send::udp_ibv_stream &stream, Callback &&callback)
    {
        heaps[next].heap.get_item(heaps[next].timestamp_handle).data.immediate = timestamp;
        timestamp += packet_samples;
        stream.async_send_heap(heaps[next].heap, callback, -1, endpoints[next_endpoint]);
        next = (next + 1) % heaps.size();
        next_endpoint = (next_endpoint + 1) % endpoints.size();
    }
};

struct digitiser
{
    spead2::thread_pool pool;
    spead2::send::udp_ibv_stream stream;
    std::vector<polarisation> pols;
    spead2::semaphore next_sem;
    std::size_t next_pol = 0;

    digitiser(const options &opts,
              std::size_t capacity,
              const std::vector<std::vector<boost::asio::ip::udp::endpoint>> &endpoints,
              const boost::asio::ip::address &interface_address)
        : pool(1),
        stream(pool, endpoints[0][0],
               spead2::send::stream_config(
                   packet_size + 128,  // Doesn't matter, just needs to be bigger than actual size
                   endpoints.size() * opts.adc_rate * 10.0 / 8.0 * (packet_size + 72) / packet_size,
                   65536, opts.max_heaps),
               interface_address),
        next_sem(opts.max_heaps)
    {
        pols.reserve(endpoints.size());
        for (const auto &ep : endpoints)
            pols.emplace_back(opts, capacity, ep);
    }

    void wait_next()
    {
        spead2::semaphore_get(next_sem);
    }

    void send_next()
    {
        auto callback = [this] (const boost::system::error_code &ec, std::size_t)
        {
            if (ec)
                std::cerr << "Error: " << ec;
            else
                next_sem.put();
        };
        pols[next_pol].send_next(stream, callback);
        next_pol++;
        if (next_pol == pols.size())
            next_pol = 0;
    }
};

int main(int argc, const char **argv)
{
    options opts = parse_options(argc, argv);

    auto interface_address = boost::asio::ip::address::from_string(opts.interface);
    std::vector<std::vector<boost::asio::ip::udp::endpoint>> endpoints;
    for (const std::string &address : opts.addresses)
        endpoints.push_back(parse_endpoint_list(address));
    digitiser d(opts, capacity, {endpoints}, interface_address);
    while (true)
    {
        d.wait_next();
        d.send_next();
    }
    return 0;
}
