#include "py_common.h"
#include "recv.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

namespace py = pybind11;

namespace katxgpu::recv
{

class py_chunk : public chunk
{
  public:
    py::buffer base;
    py::object device;
    std::shared_ptr<py::buffer_info> buffer_info;

    py_chunk(py::buffer base, py::object device)
        : base(std::move(base)), device(std::move(device)),
          buffer_info(
              std::make_shared<py::buffer_info>(request_buffer_info(this->base, PyBUF_C_CONTIGUOUS | PyBUF_WRITEABLE)))
    {
        storage = boost::asio::mutable_buffer(buffer_info->ptr, buffer_info->size * buffer_info->itemsize);
    }
};

class py_stream : public stream
{
  private:
    virtual void pre_wait_chunk() override final;
    virtual void post_wait_chunk() override final;
    virtual void pre_ringbuffer_push() override final;
    virtual void post_ringbuffer_push() override final;

  public:
    py::object monitor;

    py_stream(int n_ants, int n_channels, int n_samples_per_channel, int n_pols, int sample_bits,
              int timestamp_step, std::size_t heaps_per_fengine_per_chunk, ringbuffer_t &ringbuffer, int thread_affinity,
              bool use_gdrcopy, py::object monitor)
        : stream(n_ants, n_channels, n_samples_per_channel, n_pols, sample_bits, timestamp_step,
                 heaps_per_fengine_per_chunk, ringbuffer, thread_affinity, use_gdrcopy),
          monitor(std::move(monitor))
    {
    }
};

void py_stream::pre_wait_chunk()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
        monitor.attr("event_state")("recv", "wait free_chunk");
}

void py_stream::post_wait_chunk()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
    {
        monitor.attr("event_state")("recv", "other");
        monitor.attr("event_qsize_delta")("free_chunks", -1);
    }
}

void py_stream::pre_ringbuffer_push()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
        monitor.attr("event_state")("recv", "push ringbuffer");
}

void py_stream::post_ringbuffer_push()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
    {
        monitor.attr("event_state")("recv", "other");
        monitor.attr("event_qsize_delta")("recv_ringbuffer", 1);
    }
}

py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;

    py::module m = parent.def_submodule("recv");
    m.doc() = "receiver for katxgpu";

    py::class_<py_chunk>(m, "Chunk", "Chunk of samples")
        .def(py::init<py::buffer, py::object>(), "base"_a, "device"_a = py::none())
        .def_readwrite("timestamp", &py_chunk::timestamp)
        .def_readonly("present", &py_chunk::present)
        .def_readonly("base", &py_chunk::base)
        .def_readonly("device", &py_chunk::device);

    py::class_<py_stream>(m, "Stream", "SPEAD stream receiver")
        .def(py::init<int, int, int, int, int, int, std::size_t, stream::ringbuffer_t &, int, bool, py::object>(),
             "n_ants"_a, "n_channels"_a, "n_samples_per_channel"_a, "n_pols"_a, "sample_bits"_a,
             "timestamp_step"_a,
             "heaps_per_fengine_per_chunk"_a,
             "ringbuffer"_a, "thread_affinity"_a = -1, "use_gdrcopy"_a = false, "monitor"_a = py::none(),
             py::keep_alive<1, 6>())
        .def_property_readonly("ringbuffer",
                               [](py_stream &self) -> stream::ringbuffer_t & { return self.get_ringbuffer(); })
        .def_property_readonly("chunk_packets", &py_stream::get_chunk_packets)
        .def_property_readonly("chunk_bytes", &py_stream::get_chunk_bytes)
        .def(
            "add_chunk",
            [](py_stream &self, const py_chunk &chunk) {
                if (!self.monitor.is_none())
                    self.monitor.attr("event_qsize_delta")("free_chunks", 1);
                self.add_chunk(std::make_unique<py_chunk>(std::move(chunk)));
            },
            "chunk"_a)
        .def("add_udp_pcap_file_reader", &py_stream::add_udp_pcap_file_reader, "filename"_a)
        .def("add_udp_ibv_reader", &py_stream::add_udp_ibv_reader, "endpoints"_a, "interface_address"_a,
             "buffer_size"_a, "comp_vector"_a = 0, "max_poll"_a = spead2::recv::udp_ibv_config::default_max_poll)
        .def("stop", &py_stream::stop);

    register_ringbuffer<stream::ringbuffer_t, py_chunk>(m, "Ringbuffer", "Ringbuffer for received channelised data.");

    return m;
}

} // namespace katxgpu::recv
