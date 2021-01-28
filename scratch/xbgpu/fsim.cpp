// sudo ./fsim --interface 10.100.44.1 239.102.50.0:7148

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

struct options
{
    std::string interface;
    std::string address;
    int max_heaps = 8;
    int signal_heaps = 512;
    double adc_rate = 1712000000.0;
    int ttl = 4;
};

// TODO: These constexpr should be thought about a bit. Some of them are only used in one place and they should really
// only be calculated there.
static constexpr int n_ants = 64;     // TODO: make configurable
static constexpr int sample_bits = 8; // This is not very meaningful for the X-Engine but this argument is left here to
                                      // be consistent with the F-Engine packet simulator.
static constexpr int n_chans = 32768; // TODO: Make configurable
static constexpr int n_multicast_streams_per_antenna = 4;
static constexpr int n_xengs = n_ants * n_multicast_streams_per_antenna;
static constexpr int n_time_samples_per_channel = 256; // Hardcoded to 256 for MeerKAT
static constexpr int n_pols = 2;                       // Dual polarisation antennas
static constexpr int complexity = 2;                   // real and imaginary components

static constexpr int timestamp_step = 0x800000; // real and imaginary components

/*
 * NOTE: For the F-Engine output case, each heap is quite large but contains much smaller samples. Each packet
 * encapsualtes a single channel's worth of samples. Each packet also contains other SPEAD data and is thus slightly
 * larger than 1 KiB.
 */
static constexpr std::size_t heap_size_bytes =
    n_chans / n_xengs * n_time_samples_per_channel * n_pols * complexity * sample_bits / 8;
static constexpr std::size_t heap_samples =
    n_chans / n_xengs * n_time_samples_per_channel * n_pols; // Probably redundant
static constexpr int packet_header_size_bytes = 96;          // Nine header fields and three padding fields.
static constexpr int packet_payload_size_bytes = n_time_samples_per_channel * n_pols * complexity;
static constexpr int packet_size_bytes = packet_payload_size_bytes + packet_header_size_bytes;
static constexpr int packets_per_heap = heap_size_bytes / packet_payload_size_bytes;

// The 64 indicates that each header SPEAD2 item is 64-bits wide. The 48 value means that the ItemPointers will have 48
// bits representing the immediate value or pointer to payload. The other 16 bits will be used for the item ID.
static const spead2::flavour flavour(4, 64, 48);

// Function to assist with parsing command line parameters
template <typename T> static boost::program_options::typed_value<T> *make_opt(T &var)
{
    return boost::program_options::value<T>(&var)->default_value(var);
}

// Parse command line parameters
static options parse_options(int argc, const char **argv)
{
    options opts;
    boost::program_options::options_description desc, hidden, all;
    desc.add_options()("interface", boost::program_options::value(&opts.interface)->required(),
                       "Interface address to send data out on.");
    desc.add_options()("max-heaps", make_opt(opts.max_heaps), "Maximum number of heaps per F-Engine.");
    desc.add_options()("adc-rate", make_opt(opts.adc_rate),
                       "Sampling rate of digitisers feeding the F-Engine")("ttl", make_opt(opts.ttl), "Output TTL");
    desc.add_options()("signal-heaps", make_opt(opts.signal_heaps), "Number of pre-computed heaps to create");
    hidden.add_options()("address", boost::program_options::value<std::string>(&opts.address)->composing(),
                         "destination address, in form x.x.x.x:port");
    all.add(desc);
    all.add(hidden);
    boost::program_options::positional_options_description positional;
    positional.add("address", -1);
    try
    {
        boost::program_options::variables_map vm;
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv).options(all).positional(positional).run(), vm);
        boost::program_options::notify(vm);
        if (opts.max_heaps <= 0)
            throw boost::program_options::error("--max-heaps must be positive");
        if (opts.signal_heaps <= 0)
            throw boost::program_options::error("--signal-heaps must be positive");
    }
    catch (boost::program_options::error &e)
    {
        std::cerr << e.what() << '\n';
        std::exit(2);
    }
    return opts;
}

/* This function parses the endpoing argument passed in the program as a command line argument.
 *
 * It seperates the endpoint into its multicast ip address and port number and uses this to create an endpoint object.
 *
 * Instead of returning a single endpoint, it returns n_ants duplicates of the same endpoint. This is because the SPEAD2
 * stream object uses one substream for each antenna and each substream needs its own entry in endpoint list.
 * Duplication is the simplest way to achieve this.
 *
 * The location of this function is not too important. It could be a static method in the fengines class but has not
 * been moves as its not a priority.
 */
static std::vector<boost::asio::ip::udp::endpoint> parse_endpoint(const std::string &arg)
{
    std::vector<boost::asio::ip::udp::endpoint> out;

    auto colon = arg.find(':');
    if (colon == std::string::npos)
        throw std::invalid_argument("Address must contain a colon");

    std::uint16_t port = boost::lexical_cast<std::uint16_t>(arg.substr(colon + 1));
    for (int i = 0; i < n_ants; i++)
    {
        auto addr = boost::asio::ip::address_v4::from_string(arg.substr(0, colon));
        out.emplace_back(addr, port);
    }
    return out;
}

/* Class containing an F-Engine output heap as well as the buffer storing data pointed to by the heap. This class
 * generates simulated data to populate the heap buffer.
 */
struct heap_data
{
    std::unique_ptr<std::uint8_t[]> data;
    spead2::send::heap heap;

    // The timestamp handle is used to modify the heap timestamp after creation. This is needed as this heap will be
    // sent multiple times to reduce processing load and the timestamp needs to be updated each time it is sent.
    spead2::send::heap::item_handle timestamp_handle;

    heap_data(std::int64_t heap_index, int feng_id)
        : data(std::make_unique<std::uint8_t[]>(heap_size_bytes)), heap(flavour),
          timestamp_handle(heap.add_item(0x1600, timestamp_step * heap_index))
    {
        /* Heap format defined in section 3.4.5.2.2.1 in the "MeerKAT Functional Interface Control Document for
         * Correlator Beamformer Visibilities and Tied Array Data" (Document ID: M1000-0001-020 rev 4)
         *
         * A rough document has been put together showing the exact packet format and byte offsets produced by the
         * F-Engine: https://docs.google.com/drawings/d/1lFDS_1yBFeerARnw3YAA0LNin_24F7AWQZTJje5-XPg/edit
         */
        size_t channels_per_heap = n_chans / n_xengs;

        heap.add_item(0x4101, feng_id); // feng_id
        heap.add_item(0x4103, 32);      // frequency

        /* This field stores sample data. I need to figure out if I can set the shape of the field to have dimensions:
         * [n_chans / n_xengs][n_time_samples_per_channel][n_pols][complexity] instead of a single long dimension.
         *
         * This function adds an ItemPointer to the header and will append the data in data.get() to the packet
         * payload.
         */
        heap.add_item(0x4300, data.get(), heap_size_bytes, false); // feng_raw field

        /* The SPEAD header out of the F-Engines is aligned to 256-bit boundaries. To emulate this with SPEAD2, padding
         * needs to be added until the 256-bit boundary is reached.
         */
        for (int pad = 0; pad < 3; pad++)
            heap.add_item(0, 0);

        /* This must be set to true. It will force all immediate values to be sent in every packet instead of once per
         * heap (This applies to the 0x4101, 0x4103 and 0x1600 fields). This is needed to emulate the SKARAB F-Engines
         * as the SKARAB F-Engine duplicates these values in each packet.
         */
        heap.set_repeat_pointers(true);

        /* This section generates the sample data. A patterns is chosen that will hopefully be easy to verify at the
         * receiver graphically. On each F-Engine, the signal amplitude will increase linearly over time for each
         * channel. Each channel will have a different starting amplitude but the rate of increase will be the same for
         * all channels.
         *
         * Each F-Engine will have the same same signal amplitude for the same timestamp, but the signal phase will be
         * different. The signal phase remains constant across all channels in a single F-Engine. By examining the
         * signal phase it can be verified that correct feng_id is attached to the correct data.
         *
         * These samples need to be stored as 8 bit samples. As such, the amplitude is wrapped each time it reaches 127.
         * 127 is used as the amplitude when multiplied by the phase can reach -127. The full range of values is
         * covered.
         *
         * This current format is not fixed and it is likely that it will be adjusted to be suited for different
         * verification needs.
         */
        int initial_offset = heap_index * n_time_samples_per_channel;
        double sample_angle_pol0 = 2.0 * M_PI / ((double)(n_ants * n_pols)) * (feng_id * n_pols + 0);
        double sample_angle_pol1 = 2.0 * M_PI / ((double)(n_ants * n_pols)) * (feng_id * n_pols + 1);
        for (size_t c = 0; c < channels_per_heap; c++)
        {
            for (size_t t = 0; t < n_time_samples_per_channel; t++)
            {
                double sample_amplitude = (initial_offset + c * 10 + t) % 127;
                double sample_value_pol0_real = sample_amplitude * std::cos(sample_angle_pol0);
                double sample_value_pol0_imag = sample_amplitude * std::sin(sample_angle_pol0);
                double sample_value_pol1_real = sample_amplitude * std::cos(sample_angle_pol1);
                double sample_value_pol1_imag = sample_amplitude * std::sin(sample_angle_pol1);

                int sample_index_base = c * n_time_samples_per_channel * n_pols * complexity + t * n_pols * complexity;
                data[sample_index_base + 0] = (int8_t)sample_value_pol0_real;
                data[sample_index_base + 1] = (int8_t)sample_value_pol0_imag;
                data[sample_index_base + 2] = (int8_t)sample_value_pol1_real;
                data[sample_index_base + 3] = (int8_t)sample_value_pol1_imag;
            }
        }
    }
};

/*
 * Class to generate simulated data for a number of different F-Engines and transmit them all on a single multicast
 * address.
 *
 * SPEAD2 has the concept of substreams. Different heaps can be queued on different substreams and then the packets on
 * each heap will be interleaved. This emulates sending data from multiple sources to a single receiver with the caveat
 * that the interleaving is much more predictable than what can be expected from multiple F-Engines. One substream is
 * assigned per F-Engine. (TODO: Force interleaving by using async_send_heaps instead of async_send_heap).
 *
 * This class generates heaps for n_ants F-Engines. A heap is generated once and the transmitted multiple times so that
 * no processing time is spent creating new data.
 *
 * A number of heaps per F-Engine can be queued for flight at any one time - this allows higher transmit rates to be
 * reached. To accomodate this, each F-Engine will have max_heaps pre-generated by this class.
 */
struct fengines
{
    // This variable needs to go first so it is initialised first - it is used during the initialisation of other
    // variables.
    std::int64_t n_heaps_per_fengine;

    // SPEAD2 has its own set of threads that manage transmitting data. When SPEAD2 finishes sending a heap, it queues a
    // handler on this IO loops that is then called.
    boost::asio::io_service io_service;

    // This vector of vectors stores all the heaps. The outer vector will have one entry for each F-Engine and the inner
    // one will store the heaps per F-Engine.
    std::vector<std::vector<heap_data>> heaps;

    // SPEAD 2 stream that every heap will be queued on.
    spead2::send::udp_ibv_stream stream;

    // General variables for coordinating sending of heaps.
    std::size_t n_substreams;
    std::int64_t timestamp = 0;
    std::int64_t next_fengine = 0;
    std::int64_t next_heap = 0;

    /* Constructor for the fengines simulator.
     *
     * Initialises the SPEAD2 stream. The stream requires two main objects, the
     * stream_config() for general stream parameters and the udp_ibv_config() for more specific parameters required when
     * using ibverbs to accelerate the packet transmission.
     *
     * Creates all heaps that will queued on the SPEAD2 stream.
     */
    fengines(const options &opts, const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
             const boost::asio::ip::address &interface_address)
        : n_heaps_per_fengine(opts.max_heaps), heaps(make_heaps(opts)),
          stream(io_service,
                 spead2::send::stream_config()
                     .set_max_packet_size(packet_size_bytes)
                     .set_rate(opts.adc_rate * n_pols * sample_bits / 8.0 *
                               (heap_size_bytes + packets_per_heap * packet_header_size_bytes) / heap_size_bytes /
                               n_multicast_streams_per_antenna)
                     .set_max_heaps(opts.max_heaps * n_ants),
                 spead2::send::udp_ibv_config()
                     .set_endpoints(endpoints)
                     .set_interface_address(interface_address)
                     .set_ttl(opts.ttl)
                     .set_memory_regions(get_memory_regions(heaps)))
    {
    }

    /* Registers all heap data in memory regions and returns a vector of these regions to be used by SPEAD2.
     *
     * Memory regions are an ibverbs concept. Any collection of data that ibverbs needs to send on the network needs to
     * be in a memory region. The channel data in each heap is what is being sent in this case, and each one of these
     * needs to be added to a memory region.
     */
    static std::vector<std::pair<const void *, std::size_t>> get_memory_regions(
        const std::vector<std::vector<heap_data>> &all_heaps)
    {
        std::vector<std::pair<const void *, std::size_t>> memory_regions;
        for (const auto &single_fengine_heaps : all_heaps)
        {
            for (const auto &heap : single_fengine_heaps)
            {
                memory_regions.emplace_back(heap.data.get(), heap_size_bytes);
            }
        }
        return memory_regions;
    }

    /* Creates all the heaps required to be transmitted by the F-Engine simulator.
     *
     * This static method is a bit messy, there is probably a simpler way to do it.
     */
    static std::vector<std::vector<heap_data>> make_heaps(const options &opts)
    {
        std::vector<std::vector<heap_data>> all_fengine_heaps;
        all_fengine_heaps.reserve(n_ants);
        int heaps_per_fengine = opts.max_heaps;
        for (int feng_id = 0; feng_id < n_ants; feng_id++)
        {
            std::vector<heap_data> fengine_heaps;
            fengine_heaps.reserve(heaps_per_fengine);
            for (int heap_index = 0; heap_index < heaps_per_fengine; heap_index++)
            {
                fengine_heaps.emplace_back(heap_index, feng_id);
            }
            all_fengine_heaps.emplace_back(std::move(fengine_heaps));
        }
        return all_fengine_heaps;
    }

    /* Adds the next heap to the SPEAD2 stream queue.
     *
     * This function keeps track of the index of the next heap to send.
     *
     * It is non-blocking, once a heap has been addeded to the stream, this function will return - there is no guarentee
     * that the heap will have been sent.
     */
    void send_next()
    {
        using namespace std::placeholders;

        heaps[next_fengine][next_heap].heap.get_item(heaps[next_fengine][next_heap].timestamp_handle).data.immediate =
            timestamp;
        stream.async_send_heap(heaps[next_fengine][next_heap].heap, std::bind(&fengines::callback, this, _1, _2), -1,
                               next_fengine);

        next_fengine++;
        if (next_fengine == n_ants)
        {
            timestamp += timestamp_step;
            next_fengine = 0;
            next_heap++;
        }

        if (next_heap == n_heaps_per_fengine)
            next_heap = 0;
    }

    /* Callback function called when SPEAD2 finishes sending a heap.
     *
     * This function immediatley queues the next heap to be sent on the network.
     */
    void callback(const boost::system::error_code &ec, std::size_t)
    {
        if (ec)
        {
            std::cerr << "Error: " << ec;
            std::exit(1);
        }
        else
        {
            send_next();
        }
    }
};

/* Create fengines object and start IO loop to kick off packet tranmission.
 */
int main(int argc, const char **argv)
{
    options opts = parse_options(argc, argv);

    auto interface_address = boost::asio::ip::address::from_string(opts.interface);
    std::vector<boost::asio::ip::udp::endpoint> endpoints = parse_endpoint(opts.address);

    fengines f(opts, endpoints, interface_address);
    for (int j = 0; j < opts.max_heaps; j++)
    {
        for (int i = 0; i < n_ants; i++)
        {
        f.io_service.post(std::bind(&fengines::send_next, &f));
        }
    }

    // Will run forever.
    f.io_service.run();
    return 0;
}
