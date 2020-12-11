#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <random>
#include <boost/program_options.hpp>
#include <spead2/send_udp_ibv.h>

namespace po = boost::program_options;

struct options
{
    std::string interface;
    std::vector<std::string> addresses;
    int max_heaps = 128;
    int signal_heaps = 512;
    double adc_rate = 1712000000.0;
    double signal_freq = 232101234.0;
    int ttl = 4;
};

// Size of the heap:
// NOTE: that in this case the heap is made out of 1 KiB packets. Each packet encapsualtes a single channel and is 
// xeng_acc_length * pols * complexity = 1024 B in size
static constexpr int sample_bits = 8; // Not very meaningful for the X-Engine but this argument is left here.
static constexpr int n_chans = 32768; //TODO: Make configurable
static constexpr int n_xengs = 256; //TODO: Make configurable, generally n_ants * 4
static constexpr int xeng_acc_length = 256; //Hardcoded to 256 for MeerKAT
static constexpr int n_pols = 2; //Dual polarisation antennas
static constexpr int complexity = 2; //real and imaginary components
static constexpr std::size_t heap_size = n_chans / n_xengs * xeng_acc_length * n_pols * complexity * sample_bits / 8;
static constexpr std::size_t heap_samples = n_chans / n_xengs * xeng_acc_length * n_pols * complexity; // Probably redundant

//static constexpr int sample_bits = 8;
//static constexpr int heap_samples = 104850;
//static constexpr std::size_t heap_size = heap_samples * sample_bits / 8;

static const spead2::flavour flavour(4, 64, 48); //Not sure what this should actually be
static std::mt19937 rand_engine; //TODO: decide if this line is needed

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
        ("ttl", make_opt(opts.ttl), "Output TTL")
        ("signal-freq", make_opt(opts.signal_freq), "Frequency of simulated tone")
        ("signal-heaps", make_opt(opts.signal_heaps), "Number of pre-computed heaps to create")
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
        if (opts.max_heaps <= 0)
            throw po::error("--max-heaps must be positive");
        if (opts.signal_heaps <= 0)
            throw po::error("--signal-heaps must be positive");
        // Round target frequency to fit an integer number of waves into signal_heaps
        double waves = double(opts.signal_heaps) * heap_samples * opts.signal_freq / opts.adc_rate;
        waves = std::max(1.0, std::round(waves));
        opts.signal_freq = waves * opts.adc_rate / opts.signal_heaps / heap_samples;
        std::cout << "Using frequency of " << std::setprecision(15) << opts.signal_freq << '\n';
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

    //STEP 2: Generate heap data correctly.
    explicit heap_data(const options &opts, std::int64_t timestamp, int pol)
        : data(std::make_unique<std::uint8_t[]>(heap_size)),
        heap(flavour),
        timestamp_handle(heap.add_item(0x1600, 0))
    {
        /* Heap format defined in section 3.4.5.2.2.1 in the "MeerKAT Functional Interface Control 
         * Document for Correlator Beamformer Visibilities and Tied Array Data" 
         * (Document ID: M1000-0001-020 rev 4)
         */
        heap.add_item(0x4101, pol); //feng_id
        heap.add_item(0x4103, 0); //frequency 
        heap.add_item(0x4300, data.get(), heap_size, false); //feng_raw
        heap.set_repeat_pointers(true);

        double angle_scale = opts.signal_freq / opts.adc_rate * 2 * M_PI;
        unsigned int buffer = 0;
        int buffer_bits = 0;
        int pos = 0;
        std::uniform_real_distribution<double> noise(-0.5f, 0.5f);
        for (std::size_t i = 0; i < heap_samples; i++)
        {
            // double angle = angle_scale * (timestamp + i);
            // int sample = (int) std::round(std::sin(angle) * 256.0 + noise(rand_engine));
            // buffer = (buffer << 10) | (sample & 0x3ff);
            // buffer_bits += 10;
            // while (buffer_bits >= 8)
            // {
            //     buffer_bits -= 8;
            //     data[pos] = buffer >> buffer_bits;
            //     pos++;
            // }
        }
    }
};

struct polarisation
{
    std::vector<heap_data> heaps;
    std::size_t base_substream;
    std::size_t n_substreams;
    std::size_t next_substream = 0;   // relative to base_substream
    std::size_t next = 0;
    std::int64_t timestamp = 0;

    polarisation(const options &opts, std::size_t base_substream, std::size_t n_substreams, int pol)
        : base_substream(base_substream), n_substreams(n_substreams)
    {
        std::size_t capacity = opts.signal_heaps;
        while (capacity < std::size_t(opts.max_heaps))
            capacity *= 2;
        heaps.reserve(capacity);
        for (std::size_t i = 0; i < capacity; i++)
            heaps.emplace_back(opts, (i % opts.signal_heaps) * heap_samples, pol);
    }

    template<typename Callback>
    void send_next(spead2::send::udp_ibv_stream &stream, Callback &&callback)
    {
        heaps[next].heap.get_item(heaps[next].timestamp_handle).data.immediate = timestamp;
        timestamp += heap_samples;
        stream.async_send_heap(heaps[next].heap, callback, -1, base_substream + next_substream);
        next = (next + 1) % heaps.size();
        next_substream = (next_substream + 1) % n_substreams;
    }
};

template<typename T>
static std::vector<T> flatten(const std::vector<std::vector<T>> &in)
{
    std::vector<T> out;
    for (const auto &x : in)
        out.insert(out.end(), x.begin(), x.end());
    return out;
}

struct fengine
{
    
};

struct digitiser
{
    boost::asio::io_service io_service;
    std::vector<polarisation> pols;
    spead2::send::udp_ibv_stream stream;
    std::size_t next_pol = 0;

    static std::vector<polarisation> make_pols(
        const options &opts,
        const std::vector<std::vector<boost::asio::ip::udp::endpoint>> &endpoints)
    {
        std::vector<polarisation> pols;
        pols.reserve(endpoints.size());
        int pol = 0;
        std::size_t base_substream = 0;
        for (const auto &ep : endpoints)
        {
            pols.emplace_back(opts, base_substream, ep.size(), pol);
            pol++;
            base_substream += ep.size();
        }
        return pols;
    }

    static std::vector<std::pair<const void *, std::size_t>> get_memory_regions(
        const std::vector<polarisation> &pols)
    {
        std::vector<std::pair<const void *, std::size_t>> memory_regions;
        for (const auto &pol : pols)
            for (const auto &heap : pol.heaps)
                memory_regions.emplace_back(heap.data.get(), heap_size);
        return memory_regions;
    }

    digitiser(const options &opts,
              const std::vector<std::vector<boost::asio::ip::udp::endpoint>> &endpoints,
              const boost::asio::ip::address &interface_address)
        : pols(make_pols(opts, endpoints)),
        stream(
            io_service,
            spead2::send::stream_config()
           // eap_size = n_chans / n_xengs * xeng_acc_length * pols * complexity * sample_bits / 8;
                .set_max_packet_size(xeng_acc_length * n_pols * complexity) //STEP 1: Set this right - comment how this rate is not quite right
                .set_rate(endpoints.size() * opts.adc_rate * sample_bits / 8.0 * (heap_size + 72) / heap_size)
                .set_max_heaps(opts.max_heaps),
            spead2::send::udp_ibv_config()
                .set_endpoints(flatten(endpoints))
                .set_interface_address(interface_address)
                .set_ttl(opts.ttl)
                .set_memory_regions(get_memory_regions(pols))
        )
    {
    }

    void callback(const boost::system::error_code &ec, std::size_t)
    {
        if (ec)
        {
            std::cerr << "Error: " << ec;
            std::exit(1);
        }
        else
            send_next();
    }

    void send_next()
    {
        using namespace std::placeholders;
        pols[next_pol].send_next(stream, std::bind(&digitiser::callback, this, _1, _2));
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
    digitiser d(opts, {endpoints}, interface_address);
    for (int i = 0; i < opts.max_heaps; i++)
        d.io_service.post(std::bind(&digitiser::send_next, &d));
    d.io_service.run();
    return 0;
}
