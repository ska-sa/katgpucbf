#include <spead2/send_udp_ibv.h>

#include <boost/program_options.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace po = boost::program_options;

struct options {
    std::string interface;
    std::string address;
    int max_heaps = 128;
    int signal_heaps = 512;
    double adc_rate = 1712000000.0;
    int ttl = 4;
};

static constexpr int n_ants = 16;      // TODO: make configurable
static constexpr int sample_bits = 8;  // Not very meaningful for the X-Engine but this argument is left here.
static constexpr int n_chans = 32768;  // TODO: Make configurable
static constexpr int n_multicast_streams_per_antenna = 4;
static constexpr int n_xengs = n_ants * n_multicast_streams_per_antenna;
static constexpr int n_time_samples_per_channel = 256;  // Hardcoded to 256 for MeerKAT
static constexpr int n_pols = 2;                        // Dual polarisation antennas
static constexpr int complexity = 2;                    // real and imaginary components
/*
 * NOTE: That in this case each heap is quite large but containes much smaller samples. Each packet encapsualtes a
 * single channels worth of samples and is n_time_samples_per_channel * pols * complexity = 1024 B samples. Each packet
 * also contains other SPEAD data and is thus slightly larger than 1 KiB.
 */
static constexpr std::size_t heap_size_bytes =
    n_chans / n_xengs * n_time_samples_per_channel * n_pols * complexity * sample_bits / 8;
static constexpr std::size_t heap_samples =
    n_chans / n_xengs * n_time_samples_per_channel * n_pols;  // Probably redundant
static constexpr int timestamp_step = 0x800000;               // real and imaginary components

static const spead2::flavour flavour(4, 64, 48);  // Not sure what this should actually be

template <typename T>
static po::typed_value<T> *make_opt(T &var) {
    return po::value<T>(&var)->default_value(var);
}

static options parse_options(int argc, const char **argv) {
    options opts;
    po::options_description desc, hidden, all;
    desc.add_options()("interface", po::value(&opts.interface)->required(), "Interface address");
    desc.add_options()("max-heaps", make_opt(opts.max_heaps), "Depth of send queue (per polarisation)");
    desc.add_options()("adc-rate", make_opt(opts.adc_rate), "Sampling rate")("ttl", make_opt(opts.ttl), "Output TTL");
    desc.add_options()("signal-heaps", make_opt(opts.signal_heaps), "Number of pre-computed heaps to create");
    hidden.add_options()("address", po::value<std::string>(&opts.address)->composing(),
                         "destination address, in form x.x.x.x:port");
    all.add(desc);
    all.add(hidden);
    po::positional_options_description positional;
    positional.add("address", -1);
    try {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(all).positional(positional).run(), vm);
        po::notify(vm);
        if (opts.max_heaps <= 0) throw po::error("--max-heaps must be positive");
        if (opts.signal_heaps <= 0) throw po::error("--signal-heaps must be positive");
    } catch (po::error &e) {
        std::cerr << e.what() << '\n';
        std::exit(2);
    }
    return opts;
}

// Parse endpoint (multicast ip address and port in this case). Need to make n_ants of them as one stream per F-Engine,
// expand
static std::vector<boost::asio::ip::udp::endpoint> parse_endpoint(const std::string &arg) {
    std::vector<boost::asio::ip::udp::endpoint> out;

    auto colon = arg.find(':');
    if (colon == std::string::npos) throw std::invalid_argument("Address must contain a colon");

    std::uint16_t port = boost::lexical_cast<std::uint16_t>(arg.substr(colon + 1));
    for (int i = 0; i < n_ants; i++) {
        auto addr = boost::asio::ip::address_v4::from_string(arg.substr(0, colon));
        out.emplace_back(addr, port);
    }
    return out;
}

struct heap_data {
    std::unique_ptr<std::uint8_t[]> data;
    spead2::send::heap heap;
    spead2::send::heap::item_handle timestamp_handle;

    heap_data(std::int64_t heap_index, int feng_id)
        : data(std::make_unique<std::uint8_t[]>(heap_size_bytes)),
          heap(flavour),
          timestamp_handle(heap.add_item(0x1600, timestamp_step * heap_index)) {
        /* Heap format defined in section 3.4.5.2.2.1 in the "MeerKAT Functional Interface Control
         * Document for Correlator Beamformer Visibilities and Tied Array Data"
         * (Document ID: M1000-0001-020 rev 4)
         */
        size_t channels_per_heap = n_chans / n_xengs;

        heap.add_item(0x4101, feng_id);  // feng_id
        heap.add_item(0x4103, 0);        // frequency

        /*This field stores sample data. I need to figure out if I can set the shape of the field to have dimensions:
         * [n_chans / n_xengs][n_time_samples_per_channel][n_pols][complexity] instead of a single long string.
         */
        heap.add_item(0x4300, data.get(), heap_size_bytes, false);  // feng_raw field
        /* I think this is meant to be true. As far as I can tell, it will send the single field items in
         * every packet instead of once per heap (i.e. 0x4101, 0x4103 and 0x1600). This is needed to emulate the
         * SKARAB F-Engines as the SKARAB F-Engine duplicates these values in each packet.
         */
        heap.set_repeat_pointers(true);

        int initial_offset = heap_index * n_time_samples_per_channel;
        double sample_angle_pol0 = 2.0 * M_PI / ((double)(n_ants * n_pols)) * (feng_id * n_pols + 0);
        double sample_angle_pol1 = 2.0 * M_PI / ((double)(n_ants * n_pols)) * (feng_id * n_pols + 1);
        for (size_t c = 0; c < channels_per_heap; c++) {
            for (size_t t = 0; t < n_time_samples_per_channel; t++)  // STEP 3: Generate Simulated data.
            {
                // TODO: Document this %250 correctly
                double sample_amplitude = (initial_offset + c * 10 + t) % 125;
                double sample_value_pol0_real = sample_amplitude * std::cos(sample_angle_pol0);
                double sample_value_pol0_imag = sample_amplitude * std::sin(sample_angle_pol0);
                double sample_value_pol1_real = sample_amplitude * std::cos(sample_angle_pol1);
                double sample_value_pol1_imag = sample_amplitude * std::sin(sample_angle_pol1);

                int sample_index_base = c * n_time_samples_per_channel * n_pols * complexity + t * n_pols * complexity;
                data[sample_index_base + 0] = (uint8_t)sample_value_pol0_real;
                data[sample_index_base + 1] = (uint8_t)sample_value_pol0_imag;
                data[sample_index_base + 2] = (uint8_t)sample_value_pol1_real;
                data[sample_index_base + 3] = (uint8_t)sample_value_pol1_imag;
            }
        }
    }
};

struct fengines {
    std::int64_t n_heaps_per_fengine;
    boost::asio::io_service io_service;
    std::vector<std::vector<heap_data>> heaps;
    spead2::send::udp_ibv_stream stream;
    std::size_t n_substreams;
    std::int64_t timestamp = 0;
    std::int64_t next_fengine = 0;
    std::int64_t next_heap = 0;

    fengines(const options &opts, const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
             const boost::asio::ip::address &interface_address)
        : n_heaps_per_fengine(opts.max_heaps / n_ants),
          heaps(make_heaps(opts)),
          stream(io_service,
                 spead2::send::stream_config()
                     .set_max_packet_size(n_time_samples_per_channel * n_pols * complexity)
                     .set_rate(opts.adc_rate * n_pols * sample_bits / 8.0 * (heap_size_bytes + 72) / heap_size_bytes /
                               n_multicast_streams_per_antenna)
                     .set_max_heaps(opts.max_heaps),
                 spead2::send::udp_ibv_config()
                     .set_endpoints(endpoints)
                     .set_interface_address(interface_address)
                     .set_ttl(opts.ttl)
                     .set_memory_regions(get_memory_regions(heaps))) {}

    static std::vector<std::pair<const void *, std::size_t>> get_memory_regions(
        const std::vector<std::vector<heap_data>> &all_heaps) {
        std::vector<std::pair<const void *, std::size_t>> memory_regions;
        for (const auto &single_fengine_heaps : all_heaps) {
            for (const auto &heap : single_fengine_heaps) {
                memory_regions.emplace_back(heap.data.get(), heap_size_bytes);
            }
        }
        return memory_regions;
    }

    static std::vector<std::vector<heap_data>> make_heaps(const options &opts) {
        std::vector<std::vector<heap_data>> all_fengine_heaps;
        all_fengine_heaps.reserve(n_ants);
        int heaps_per_fengine = opts.max_heaps / n_ants;  // TODO: Neaten this up a bit
        for (int feng_id = 0; feng_id < n_ants; feng_id++) {
            std::vector<heap_data> fengine_heaps;
            fengine_heaps.reserve(heaps_per_fengine);
            for (int heap_index = 0; heap_index < heaps_per_fengine; heap_index++) {
                fengine_heaps.emplace_back(heap_index, feng_id);
            }
            all_fengine_heaps.emplace_back(std::move(fengine_heaps));
        }
        return all_fengine_heaps;
    }

    void send_next() {
        using namespace std::placeholders;

        heaps[next_fengine][next_heap].heap.get_item(heaps[next_fengine][next_heap].timestamp_handle).data.immediate =
            timestamp;
        stream.async_send_heap(heaps[next_fengine][next_heap].heap, std::bind(&fengines::callback, this, _1, _2), -1,
                               next_fengine);

        next_fengine++;
        if (next_fengine == n_ants) {
            timestamp += timestamp_step;
            next_fengine = 0;
            next_heap++;
        }

        if (next_heap == n_heaps_per_fengine) next_heap = 0;
    }

    void callback(const boost::system::error_code &ec, std::size_t) {
        if (ec) {
            std::cerr << "Error: " << ec;
            std::exit(1);
        } else
            send_next();
    }
};

int main(int argc, const char **argv) {
    options opts = parse_options(argc, argv);

    auto interface_address = boost::asio::ip::address::from_string(opts.interface);
    std::vector<boost::asio::ip::udp::endpoint> endpoints = parse_endpoint(opts.address);

    fengines f(opts, endpoints, interface_address);
    for (int j = 0; j < opts.max_heaps / n_ants; j++) {
        for (int i = 0; i < n_ants; i++) {
            f.io_service.post(std::bind(&fengines::send_next, &f));
        }
    }
    f.io_service.run();
    return 0;
}
