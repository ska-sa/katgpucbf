Operation
=========
There are two main scenarios involved in starting up and interacting with
katgpucbf and its constituent engines:

#. the instantiation and running of a complete end-to-end correlator, and
#. the invocation of individual engines (dsim, fgpu, xbgpu) for more
   fine-grained testing and debugging.

The first requires a mechanism to orchestrate the simultaneous spin-up of a
correlator's required components - that is, some combination of dsim(s),
F-Engine(s) and XB-Engine(s). For this purpose, katgpucbf utilises the
infrastructure provided by `katsdpcontroller`_ — discussed in the following
section.

Regarding the testing and debugging of individual engines, more detailed
explanations of their inner workings are discussed in their respective, more
dedicated discussion documents.

The main thing to note is that, in both methods of invocation (via
orchestration and individually), the engines support control via katcp requests
issued to their ``<host>:<port>``. ``netcat`` (`nc`_) is likely the most
readily-available tool for this job, but `ntsh`_ neatens up these exchanges
and generally makes it easier to interact with.

.. _katsdpcontroller: https://github.com/ska-sa/katsdpcontroller
.. _nc: https://www.commandlinux.com/man-page/man1/nc.1.html
.. _ntsh: https://pypi.org/project/ntsh/

.. _katsdpcontroller-discussion:

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
The katgpucbf repository comes with a ``scratch/`` directory, under which you
will find handy scripts for correlator and engine invocation. Granted, the
layout and usage of these scripts is tailored to SARAO DSP's internal lab
development environment (e.g. host and interface names) and don't necessarily
go through the same reviewing rigour as the actual codebase. For these reasons,
it is recommended that these scripts are used more as an example of how to run
components of katgpucbf, rather than set-in-stone modi operandi.

End-to-end correlator startup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you intend on starting up a correlator with :program:`sim_correlator.py`,
you will require a running master controller in accordance with
:ref:`katsdpcontroller-discussion`. The script itself provides an array of
options for you to start your correlator; running ``./sim_correlator.py --help``
gives a brief explanation of the arguments required. Below is an example of a
full command to run a 4k, 4-antenna, L-band correlator::

    scratch/sim_correlator.py -a 4 -c 4096 -i 0.5 --band l \
        --name my_test_correlator lab5.sdp.kat.ac.za

The execution of this command contacts the master controller to request a new
correlator product to be configured. The master controller figures out how many
of each respective engine is required based on these input parameters, and
launches them accordingly across the pool of processing nodes available.

.. _indiv-engine-startup:

Individual engine startup
^^^^^^^^^^^^^^^^^^^^^^^^^
The arguments required for individual engine invocation can be seen by
running one of ``{dsim, fgpu, xbgpu} --help`` in an appropriately-configured
terminal environment. There are a few mandatory ones, and ultimately stitching
the entire incantation together by hand can become tiresome. For this reason,
the scripts under ``scratch/{fgpu, xbgpu}`` have been shipped with the module.

The scripts for standalone engine usage are prepopulated with typical
configuration values for your convenience, and are usually named
:program:`run-{dsim, fgpu, xbgpu}.sh`. It is important to note that the F- and
XB-Engines can run in a standalone manner, but will require some form of
stimulus to truly exercise the engine. For example, ``fgpu`` requires a
corresponding ``dsim`` to produce data for ingest. Similarly, ``xbgpu``
requires an appropriately-configured ``fsim``. Basically, the engines will do
nothing until explicitly asked to.

.. todo:: ``NGC-730``
  Add comments on the scripts themselves to make them easier to follow.

.. note::
    Before considering which engine you intend on testing, note the number of GPUs
    available in the target processing node. The `CUDA`_ library acknowledges the
    presence of a ``CUDA_VISIBLE_DEVICES`` environment variable, similar to that
    discussed by :external+katsdpsigproc:std:ref:`katsdpsigproc <configuration>`.
    You can simply ``export CUDA_VISIBLE_DEVICES=0`` in your terminal environment
    for the engine invocation to acknowledge your intention of using a particular
    GPU.

.. _CUDA: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars

To test a 4k, 4-antenna XB-Engine processing L-band data, use the following
commands in separate terminals on two separate servers. This will launch a
single :ref:`feng-packet-sim` on ``host1`` and a single :program:`xbgpu`
instance on ``host2``::

    [Connect to host1 and activate the local virtual environment]
    (katgpucbf) user@host1:~/katgpucbf$ spead2_net_raw fsim --interface <interface name> --ibv \
                                        --array-size 4 --channels 4096 \
                                        --channels-per-substream 1024 \
                                        239.10.10.10+1:7148
    .
    .
    .
    [Connect to host2 and activate the local virtual environment]
    (katgpucbf) user@host2:~/katgpucbf$ spead2_net_raw numactl -C 1 xbgpu \
                                        --recv-affinity 0 --recv-comp-vector 0 \
                                        --send-affinity 1 --send-comp-vector 1 \
                                        --recv-interface <interface name> \
                                        --send-interface <interface name> \
                                        --recv-ibv --send-ibv \
                                        --adc-sample-rate 1712e6 --array-size 4 \
                                        --channels 4096 \
                                        --channels-per-substream 1024 \
                                        --samples-between-spectra 8192 \
                                        --katcp-port 7150 \
                                        239.10.10.10:7148 239.10.11.10:7148

Naturally, it is up to the user to ensure command-line parameters are
consistent across the components under test, e.g. using the same
:option:`!--array-size` is for the data generated (in the :program:`fsim`) and
the :program:`xbgpu` instance.

.. note::
    ibverbs requires ``CAP_NET_RAW`` capability on Linux hosts. See
    :external+spead2:std:ref:`spead2's discussion <spead2_net_raw>` on
    ensuring this is configured correctly for your usage.

Pinning thread affinities
"""""""""""""""""""""""""
.. todo:: ``NGC-730``
  Update ``run-{dsim, fpgu, xbgpu}.sh`` scripts to standardise over usage
  of either ``numactl`` or ``taskset``.

:external+spead2:doc:`spead2's performance tuning discussion <perf>` outlines
the need to set the affinity of all threads that aren't specifically pinned by
:option:`!--{src, dst}-affinity`. This is often the main Python thread, but
libraries like CUDA tend to spin up helper threads.

Testing without a high-speed data network
"""""""""""""""""""""""""""""""""""""""""
katgpucbf allows the user to develop, debug and test its engines without the
use of a high-speed e.g. 100GbE data network. The omission of
:option:`!--{src, dst}-ibv` command-line parameters avoids receiving data via
the Infiniband Verbs API. This means that if you wish to e.g. capture engine
data on a machine that doesn't support ibverbs, you could use
:manpage:`tcpdump(8)`.

.. note::
    The data rates you intend to process are still limited by the NIC in your
    host machine. To truly take advantage of running engines without a
    high-speed data network, consider reducing the :option:`!--adc-sample-rate`
    by e.g. a factor of ten as this value greatly affects the engine's data
    transmission rate.

Controlling the correlator
--------------------------
The correlator components are controlled using `katcp`_. A user can connect to
the ``<host>:<port>`` and issue a ``?help`` to see the full range of requests
available. The ``<host>`` and ``<port>`` values for individual engines are
configurable at runtime, whereas the ``<host>`` and ``<port>`` values for the
correlator's *product controller* are yielded by the master controller after
startup. Standard katcp requests (such as querying and subscribing to sensors)
are not covered here; only application-specific requests are listed. Sensors
are described in :ref:`monitoring-sensors`.

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
    the different streams. Repeating the same request thus gives the same
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
:samp:`?gain {stream} {input} [{values}...]`
    Set the complex gains. This has the same semantics as the equivalent
    katsdpcontroller request, but :samp:`{input}` must be 0 or 1 to select
    the input polarisation.

:samp:`?gain-all {stream} {values}...`
    Set the complex gains for both inputs. This has the same semantics as the
    equivalent katsdpcontroller request.

:samp:`?delays {stream} {start-time} {values}...`
    Set the delay polynomials. This has the same semantics as the equivalent
    katsdpcontroller request, but takes exactly two delay model
    specifications (for the two polarisations).

xbgpu
^^^^^
:samp:`?capture-start {stream} [{timestamp}]`, :samp:`?capture-stop {stream}`
    Enable or disable transmission of output data. This does not affect
    transmission of descriptors, which cannot be disabled. In the initial
    state transmission is disabled, unless the :option:`!--send-enabled`
    command-line option has been passed.

    If :samp:`{timestamp}` is specified, heaps with timestamps less than this
    ADC timestamp will not be transmitted.

:samp:`?beam-weights {stream} {weights}...`, :samp:`?beam-delays {stream} {delays}...`, :samp:`?beam-quant-gains {stream} {gain}`
    These have the same semantics as the equivalent katsdpcontroller
    requests.

Shutting down the correlator
----------------------------

End-to-end correlator shutdown
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A user can issue a ``?product-deconfigure`` request to the correlator's
product controller by connecting to its ``<host>:<port>``. This request
triggers the stop procedure of all engines and dsims running in the target
correlator. More specifically:

* the product controller instructs the orchestration software to stop the
  containers running the engines,
* which is received by the engines as a ``SIGTERM``,
* finally triggering a ``halt`` in the engines for a graceful shutdown.

The shutdown procedures are broadly similar between the dsim, fgpu and xbgpu.
Ultimately they all:

* finish calculations on data currently in their pipelines,
* stop the transmission of their SPEAD descriptors, and
* in the case of ``fgpu`` and ``xbgpu``, stop their ``spead2`` receivers, which
  allows for a more natural ending of internal processing operations.

Individual engine shutdown
^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you've sufficiently tested, debugged and/or reached the desired level of
confusion, there are two options for engine shutdown:

#. simply issue a ``Ctrl + C`` in the terminal window where the engine was
   invoked, or
#. connect to the engine's ``<host>:<port>`` and issue a ``?halt``.

After either of these approaches are executed, the engine will shutdown cleanly
and quietly according to their common :ref:`engines-shutdown-procedure`.
