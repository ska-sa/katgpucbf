.. _fsim:

F-Engine Packet Simulator
-------------------------

In general an X-Engine needs to receive data for a subset of channels
from N F-Engines where N is the telescope array size. This is complicated to
configure and requires many F-Engines. In order to bypass this, an F-Engine
simulator has been created that simulates packets received at the X-Engine (i.e.,
packets from multiple F-Engines destined for the same X-Engine). This simulator
requires a server with a Mellanox NIC and ibverbs to run. This fsim simulates
the exact packet format from the SKARAB F-Engines. The SKARAB X-Engines ingest
data from 4 different multicast streams.

To build the fsim, navigate to the ``src/tools/`` directory and run ``make``.

The minimum command to run the fsim is:

.. code-block:: bash

    ./fsim --interface <interface_address> <multicast_address>[+y]:<port>

Where:

- `<interface_address>` is the ip address of the network interface to transmit
  the data out on.
- `<multicast_address>` is the multicast address all packets are destined to.
  The optional `[+y]` argument will create `y` additional multicast streams
  with the same parameters each on a different multicast address
  consecutively after the base `<multicast_address>`.
- `<port>` is the UDP port to transmit data to.

The above command will transmit data at about `7.8 * (y+1)` Gbps by default.

The ``fsim`` executable needs ``CAP_NET_RAW`` capability to run. The easiest
way to do it is with ``spead2_net_raw``.

See the fsim source code (in ``src/tools/``) for a  detailed description of how the
F-Engine simulator works and the useful configuration arguments.
