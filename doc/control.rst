Operation
=========
There are two main scenarios involved in starting up and interacting with
katgpucbf and its constituent engines:

#. the instantiation and running of a complete end-to-end correlator, and
#. the invocation of individual engines (dsim, fgpu, xbgpu) for more
   fine-grained testing and debugging.

The first of which requires a mechanism to orchestrate the simultaneous spin-up
of a correlator's required components - that is, some combination of dsim(s),
F-Engine(s) and XB-Engine(s). For this purpose, katgpucbf utilises the
infrastructure provided by `katsdpcontroller`_ - discussed in the following
section.

Regarding the testing and debugging of individual engines, more detailed
explanations of their inner-workings are discussed in their respective
documents. An example of this scenario is running a standalone instance of
``xbgpu`` along with an appropriately-configured ``fsim``.

* Here, you might use one of the handy scripts under e.g. ``scratch/xbgpu/``
  to launch an XB-Engine instance.


The main thing to note for the context of this document is that, in
both methods of invocation (via orchestration and individually), the engines
support control via katcp commands issued to their ``<host>:<port>`` via a
line-based networking utility. ``netcat`` (`nc`_) is likely the most
readily-available tool for this job, but `ntsh`_ neatens up these exchanges
and generally makes it easier to interact with.

.. _katsdpcontroller: https://github.com/ska-sa/katsdpcontroller
.. _nc: https://www.commandlinux.com/man-page/man1/nc.1.html
.. _ntsh: https://pypi.org/project/ntsh/

katsdpcontroller
----------------
This package (katgpucbf) provides the components of a correlator (engines and
simulators), but not the mechanisms to start up and orchestrate all the
components as a cohesive unit. That is provided by `katsdpcontroller`_.

For production use it is strongly recommended that katsdpcontroller is used to
manage the correlator. Nevertheless, it is possible to run the individual
pieces manually, or to implement an alternative controller. The remaining
sections in this chapter describe the interfaces that are used by
katsdpcontroller to communicate with the correlator components.

There are two parts to katsdpcontroller: a :dfn:`master controller` and a
:dfn:`product controller`. There is a single product controller per
instantiated correlator. It is responsible for:

- starting up the appropriate correlator components with suitable arguments,
  given a high-level description of the desired correlator configuration;
- monitoring the health of those components;
- registering them with `Consul`_, so that infrastructure such as `Prometheus`_
  can discover them;
- proxying their :ref:`monitoring-sensors`, so that clients need only
  subscribe to sensors from the product controller rather than individual
  components;
- in some cases, aggregating or renaming those sensors, to present a
  correlator-wide suite of sensors, without clients needing to know about the
  individual engines;
- providing additional correlator-wide katcp sensors;
- providing correlator-wide katcp requests, which are implemented by issuing
  similar but finer-grained requests to the individual engines.

.. _Consul: https://www.consul.io/
.. _Prometheus: https://prometheus.io/

The master controller manages product controllers (and hence correlators),
starting them up and shutting them down on request from the user. In a system
supporting subarrays, there will typically be a single master controller and
zero or more product controllers at any one time.

It is worth noting that katsdpcontroller was originally written to control the
MeerKAT Science Data Processor and later extended to control correlators, so
it has a number of features, requests and sensors that are not relevant to
correlators.

Starting the correlator
-----------------------

.. todo::  ``NGC-684``
    Describe how a correlator should be started. Master controller figures out
    based on a set of input parameters, how to invoke a few instances of
    katgpucbf as dsim, fgpu or xbgpu.


Controlling the correlator
--------------------------

The correlator components are controlled using `katcp`_. Standard katcp
requests (such as querying and subscribing to sensors) are not covered here;
only application-specific requests are listed. Sensors are described in
:ref:`monitoring-sensors`.

.. _katcp: https://katcp-python.readthedocs.io/en/latest/_downloads/361189acb383a294be20d6c10c257cb4/NRF-KAT7-6.0-IFCE-002-Rev5-1.pdf

dsim
^^^^
:samp:`?signals {spec} [{period}]`
    Change the signals that are generated. The signal specification is
    described in :ref:`dsim-dsl`. The resulting signal will be periodic with a
    period of :samp:`{period}` samples. The given period must divide into the
    :option:`!--max-period` command-line argument, which is also the default
    period if none is specified.

    The dither that is applied is cached on startup, but is independent for
    the different streams. Repeating the same command thus gives the same
    results, provided any randomised terms (such as ``wgn``) use fixed
    seeds.

    It returns an ADC timestamp, which indicates the next sample which is
    generated with the new signals. This is kept for backwards compatibility,
    but the same information can be found in the ``steady-state-timestamp``
    sensor.

``?time``
    Return the current UNIX timestamp on the server running the dsim. This can
    be used to get an approximate idea of which data is in flight, without
    depending on the dsim host and the client having synchronised clocks.

fgpu
^^^^
:samp:`?gain {input} [{values}...]`
    Set the complex gains. This has the same semantics as the equivalent
    katsdpcontroller command, but :samp:`{input}` must be 0 or 1 to select
    the input polarisation.

:samp:`?gain-all {values}...`
    Set the complex gains for both inputs. This has the same semantics as the
    equivalent katsdpcontroller command.

:samp:`?delays {start-time} {values}...`
    Set the delay polynomials. This has the same semantics as the equivalent
    katsdpcontroller command, but takes exactly two delay model
    specifications (for the two polarisations).

xbgpu
^^^^^
``?capture-start``, ``?capture-stop``
    Enable or disable transmission of output data. This does not affect
    transmission of descriptors, which cannot be disabled. In the initial
    state transmission is enabled.

    .. todo:: Update after NGC-721 is addressed

Shutting down the correlator
----------------------------

Normal correlator operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
A user can connect to the correlator's ``<ip_addr>:<port>`` using a line-based
networking utility and issue a ``?product-deconfigure`` command. This
``?product-deconfigure`` command triggers the stop procedure of all
engines and dsims running in the target correlator. The dsim, fgpu and xbgpu
all make use of the
:external+aiokatcp:py:class:`aiokatcp server <aiokatcp.server.DeviceServer>`'s
:external+aiokatcp:py:meth:`on_stop <aiokatcp.server.DeviceServer.on_stop>`
feature which allows for any engine-specific clean-up to take place before
coming to a final halt.

The ``on_stop`` procedure is vastly similar between the dsim, fgpu and xbgpu.

* The ``dsim`` simply stops its internal calculation and sending processes of
  data and descriptors respectively.
* ``fgpu`` and ``xbgpu`` both stop their respective
  :external+spead2:doc:`spead2 receivers <recv-chunk>`, which allows for a more
  natural ending of internal processing operations.

  *  Each stage of processing passes a `None`-type on to the next stage,
  *  Eventually resulting in the engine sending a
     :external+spead2:doc:`SPEAD stop heap <py-protocol>` across its output
     streams.

Engine-specific testing and debugging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you've sufficiently debugged and/or reached the desired level of confusion,
you can simply issue a ``Ctrl + C`` in your terminal window. ``xbgpu`` will shut
down cleanly and quietly according to the stop procedure mentioned above.

A fair bit of work has gone into ensuring the engines and
:external+aiokatcp:py:class:`DeviceServers <aiokatcp.server.DeviceServer>`
they're built on are robust to a variety of exceptions and anomalies. Adding to
that, the reporting of errors and exceptions has been consolidated for ease of
traceability, e.g. according to each stage of the processing chain (receive,
gpu-processing, transmit). This reduces the potential chaos involved in
monitoring correlator-wide operations.
