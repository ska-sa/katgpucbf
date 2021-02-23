# Network Interface Code and SPEAD2

This README attempts to give an overview of the different moving parts of transmit and receive code for the katxgpu
program. This explanation attempts to be as complete as possible, but there are many moving parts to this system so 
missing information is likely. 

This overview takes place within the context of the MeerKAT telescope as such the MeerKAT data formats will be
discussed.

## 1. Key Concepts

Below are some of the key concepts involved:

### 1.1 SPEAD

The Streaming Protocol for Exchanging Astronomical Data (SPEAD) is a protocol for transmitting radio astronomy data 
over a network at high data rates. SPEAD functions as a layer on top of UDP. The documentation for SPEAD can be found
[here](https://casper.ssl.berkeley.edu/wiki/SPEAD).

SPEAD transmits logical collections of data known as heaps. A heap consists of multiple UDP packets. A SPEAD 
transmitter will decompose a heap into packets and the receiver will collect all the packets and reassemble the heap.

The heaps and corresponding packet formats received by katxgpu have already been defined. This 
[document](https://docs.google.com/drawings/d/1lFDS_1yBFeerARnw3YAA0LNin_24F7AWQZTJje5-XPg) explains these formats in 
detail.

### 1.2 SPEAD2

SPEAD2 is a software package that implements the SPEAD protocol. It can be used to both send and receive SPEAD heaps.
SPEAD2 is designed to be very high performance, able to receive and transmit data losslessly at 100 GbE data rates when
used correctly.

SPEAD2 has both Python and C++ bindings. Python is the conventional way to use SPEAD2 with C++ being used to implement 
additional features for high performance needs.

The documentation for SPEAD2 can be found [here](https://spead2.readthedocs.io/en/latest/). An index of all the 
different SPEAD2 functions can be found [here](https://spead2.readthedocs.io/en/latest/genindex.html).

#### 1.2.1 SPEAD2 transports

SPEAD2 has the concept of a "transport". A transport can be thought of the interface with the underlying system that the
SPEAD heaps are transmitted on. The most basic example of a transport is a UDP transport. SPEAD2 with a UDP transport
will make use of the Linux networking stack to receive packets off of an ethernet interface. 

SPEAD2 supports a number of different transports including a udp transport for standard UDP networking, a udp_ibv
transport (explained in 1.2.2 below), a PCAP transport for reading data from a PCAP file, a buffer transport for reading
simulating packets in memory and others. 

The advantage of having these different transports is that the interface from the main program to SPEAD2 can be
decoupled from the underlying transport used. If a new high performance library becomes available for transmitting
network data, it can be added to SPEAD2 without the user having to change their interface as SPEAD2. SPEAD2 will just
transfer a completed heap to the main program no matter the transport being used.

In this repository, the main example of where these transports are useful is when unit testing. During normal
operation, the udp_ibv transport is used for high performance receiving of packets off of the ethernet network. It is not practical to run unit tests on the network. When performing unit tests, a buffer transport is used instead and SPEAD2 assembles simulated
packets from a memory buffer into heaps, thereby testing the SPEAD2 functionality without having to be connected to the
network.

The only intervention required by the user is to tell SPEAD2 what transport to use. When receiving data, this is done
using functions such as `add_udp_ibv_reader`, `add_udp_pcap_reader` or `add_memory_reader`.

TODO: List the functions required to specify what transport to use for transmitting data when the tranmit code is added.

#### 1.2.2 ibverbs

SPEAD2 implements a transport using the ibverbs library for high performance networking. This is the udp_ibv transport.
Using ibverbs for ethernet NIC acceleration is not very well documented online. SARAO has produced this 
[ibverbs sample project](https://github.com/ska-sa/dc_sand/tree/master/ibverbs_sample_project) to demonstrate how to 
use ibverbs and explain how it functions. A deep understanding of ibverbs is not required here as SPEAD2 handles all
of the complexity. 

Ibverbs requires Mellanox ConnectX NICs and the Mellanox OFED drivers to be installed in order to function. This is
explained in more detail in the top level [README](../README.md).

#### 1.2.3 Asyncio

When SPEAD2 is run, it launches its own threads. These threads interact with the main program using an asyncio loop. 
When a heap is received or sent, SPEAD2 puts an event on a specified IO loop indicating that this action has occured.

### 1.3 Multicast

All SARAO SPEAD traffic is transmitted as ethernet multicast data. Ethernet multicast is not as simple as unicast. In
general the switches need to be configured to handle multicast data (using an implementation of the PIM protocol for L3
or the IGMP protocol for L2 networks.). A receiver also needs to subscribe to multicast data in order for the network
to transmit it to the receiver. SPEAD2 handles issuing the subscription on the server, the network needs to be
configured to propegate these subscriptions correctly. Ethernet routes stored in the server OS need to be correctly
configured to ensure multicast trafficis being received or transmitted through the correct interface.

If data is not being transmitted or received correctly, it is best to first ensure that multicast traffic is being
routed correctly on the network.

### 1.4 Chunks

In order to reduce the load on the slow Python controlling code, multiple SPEAD heaps are combined into a single chunk
in the high performance C++ code before being passed to Python. Python then launches GPU kernels to operate on a single 
chunk.

An example of why this is necessary: a single F-Engine output heap is 0.125 MiB. At 7 Gbps, ~60 000 heaps are passed to
python every second. This is a very high load on the CPU and results in the python thread not being able to keep up. A
single chunk consists of a collection of 10 heaps from every antenna for a chunk size of 10x64x0.125=80MiB. At 7 Gbps,
~90 chunks are passed to python per second. This is a much more manageable number of chunks for slow Python code to deal
with.

Additionally, executing a GPU kernel on a large chunk instead of a single heap allows the kernel to be launched with
many more threads meaning far better utilisation of the GPU takes place.

### 1.5 Low level C++ code and python bindings

katxgpu does not make use of the SPEAD2 python bindings. Instead it makes use of the lower level C++ SPEAD2 functions 
and then exposes them to python with its own bindings. This was done because ordinarily SPEAD2 does not have the concept
of a chunk. Using C++ to implement SPEAD2 allows these heaps to be assembled into chunks before transferring them to
python.

The C++ code can be quite dense and complicated. Much effort has been put into making katxgpu readable and functional
without having to delve into the C++ code.

The [pybind11 library](https://pybind11.readthedocs.io/en/stable/index.html) is used for registering C++ code as a
python module. The C++ files doing this can be found in the [katxgpu/src](.) directory. The [setup.py](../setup.py) file
handles turning these C++ files into python modules. The [py_register.cpp](./py_register.cpp) contains the
`PYBIND11_MODULE` macro which kicks off the process during installation.

## 2. Receiver

![Receiver](./katxgpu_receiver.png)

### 2.1 Top level view

bindings can be found [py_recv.cpp](./py_recv.cpp) and [py_recv.h](./py_recv.h)

logic can be found [recv.cpp](./recv.cpp) and [py_recv.h](./recv.h)

Example can be found here: [example](../scratch/receiver_example.py)

Discuss user side functionality

### 2.2 Chunk Lifecycle

### 2.3 Allocating heaps to chunks

#### 2.3.1 Timestamp Alignment

#### 2.3.2 Data layout in a chunk

[heaps_per_fengine_per_chunk][n_ants][n_channels_per_stream][n_samples_per_channel][n_pols]

### 2.4 Transport and readers

Only a few readers have been implemented yet

## 3. Sender

Sender logic still needs to be implemented. This section will be updated once this has occured.

## 4. Peerdirect Support

TODO: Write a script demonstrating how to use Peerdirect support

## 5. Testing

### Unit Tests

`spead2.send.InprocStream` object to feed the `inproc_queue` added to the   `add_inproc_reader`
 [receiver unit test](../test/spead2_receiver_test.py)

### Receiveing from a pcap file
