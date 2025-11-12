System requirements
===================
For basic operation (such as for development or proof-of-concept) the only
hardware requirement is an NVIDIA GPU with tensor cores. The rest of this
section describes recommended setup for high-performance operation.

Networking
----------
An NVIDIA NIC (ConnectX or Bluefield) should be used, as katgpucbf can bypass
the kernel networking stack when using one of these NICs. See the spead2
:external+spead2:doc:`documentation <py-ibverbs>` for details on setting up and
tuning the ibverbs support. Pay particular attention to disabling multicast
loopback.

The correlator uses multicast packets to communicate between the individual
engines. Your network needs to be set up to handle multicast, and to do so
efficiently (i.e., not falling back to broadcasting). Note that the
out-of-the-box configuration for Spectrum switches running Onyx allocates very
little buffer space to multicast traffic, which can easily lead to lost
packets. Refer to the manual for your switch to adjust the buffer allocations.

The engines also default to using large packets (8 KiB of payload, plus some
headers), so your network needs to be configured to support jumbo frames. While
there are command-line options to reduce the packet sizes, this will
significantly reduce performance.

BIOS settings
-------------
See the system tuning guidance in the :external+spead2:doc:`spead2
documentation <perf>` and in [Merry2023]_.
In particular, we've found that when running multiple
F-engines per host on an AMD Epyc (Milan) system, we get best performance with

- NPS1 setting for NUMA per socket (NPS2 might work too, but NPS4 tends to
  cause sporadic lost packets);
- the GPU and the NIC in slots attached to different host bridges.
