#ifndef KATXGPU_RECV_H
#define KATXGPU_RECV_H

#include <boost/asio.hpp>
#include <vector>

namespace katxgpu::recv
{

/* Most of therse are irrelivant, must neaten then up */
static constexpr int n_ants = 64;     // TODO: make configurable
static constexpr int sample_bits = 8; // Not very meaningful for the X-Engine but this argument is left here.
static constexpr int n_chans = 32768; // TODO: Make configurable
static constexpr int n_multicast_streams_per_antenna = 4;
static constexpr int n_xengs = n_ants * n_multicast_streams_per_antenna;
static constexpr int n_time_samples_per_channel = 256; // Hardcoded to 256 for MeerKAT
static constexpr int n_pols = 2;                       // Dual polarisation antennas
static constexpr int complexity = 2;                   // real and imaginary components

/**
 * TODO: Figure out what this should actually store.
 */
struct chunk
{
    std::int64_t timestamp;              ///< Timestamp of first sample
    std::vector<bool> present;           ///< Bitmask of packets that are present
    boost::asio::mutable_buffer storage; ///< Storage for samples

    virtual ~chunk() = default; // makes it polymorphic
};

} // namespace katxgpu::recv

#endif // KATFGPU_RECV_H