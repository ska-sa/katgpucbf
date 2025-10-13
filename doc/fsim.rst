.. _feng-packet-sim:

F-Engine Packet Simulator
-------------------------

In general an XB-Engine needs to receive data for a subset of channels
from N F-Engines where N is the telescope array size. This is complicated to
configure and requires many F-Engines. In order to bypass this, an F-Engine
simulator has been created that simulates packets received at the XB-Engine (i.e.,
packets from multiple F-Engines destined for the same XB-Engine). This simulator
benefits from a server with a Mellanox NIC and ibverbs to run. This fsim
simulates the packet format used by katgpucbf.

The minimum command to run fsim is:

.. code-block:: sh

    fsim --interface <interface_name> --adc-sample-rate <rate> <multicast_address>[+y]:<port>

where

- `<interface_name>` is the name of the network interface on which to transmit the data;
- `<multicast_address>` is the multicast address to which all packets are sent.
  The optional `[+y]` argument will create additional multicast streams with
  the same parameters each on a different multicast addresses consecutivly
  after the base address. `<port>` is the UDP port to transmit data to.

The data rate per multicast address is
adc_rate * N_POLS * SAMPLE_BITS * antennas * (channels_per_substream /
channels). With the default arguments, this is
1712000000 * 2 * 8 * 80 * (512/32768) = 34.24 Gbps.

For improved performance, use :option:`!--ibv` to enable ibverbs acceleration [#]_.
This requires the ``CAP_NET_RAW`` capability to run. The easiest way to do it
is with ``spead2_net_raw``.

See the fsim source code (in ``src/katgpucbf/fsim/``) for a  detailed
description of how the F-Engine simulator works and the useful configuration
arguments.

.. [#] See the spead2 documentation for information on the requirements
   (particularly hardware requirements) for ibverbs.
