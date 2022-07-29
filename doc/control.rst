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
explanations of their inner-workings are discussed in their respective, more
dedicated-discussion documents.

The main thing to note for the context of this document is that, in
both methods of invocation (via orchestration and individually), the engines
support control via katcp commands issued to their ``<host>:<port>`` (using a
line-based networking utility). ``netcat`` (`nc`_) is likely the most
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
The katgpucbf repository comes with a ``scratch/`` directory, under which you
will find handy scripts for correlator and engine invocation.

End-to-end correlator startup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:program:`sim_correlator.py` provides an array of options for you to start your
required correlator. Running ``./sim_correlator.py --help`` gives a brief
explanation of the arguments required. Below is an example of a full command::

    ./sim_correlator -a 2 -c 4096 -i 0.5
    --adc-sample-rate 171e6
    --name my_test_correlator
    --image-override katgpucbf:harbor.sdp.kat.ac.za/cbf/katgpucbf:latest
    lab5.sdp.kat.ac.za

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

Adding to this, the :file:`scratch/{fgpu, xbgpu}` directory contains a final
nesting doll named ``config`` containing configuration information for
commonly-used hosts when invoking engines manually. These simply include
100GbE interfaces available to use, and the

dsim
""""


fgpu
""""


xbgpu
"""""
Here, you might use one of the handy scripts under ``scratch/xbgpu/`` to
launch an XB-Engine instance and a corresponding :ref:`feng-packet-sim` using
:program:`run-xbgpu.sh` and :program:`run-fsim.sh`.

Controlling the correlator
--------------------------
A timely reminder for the context of this document regarding correlator and
engine interactions:

* the ``<host>`` and ``<port>`` values for individual engines are configurable at
  runtime, whereas
* the ``<host>`` and ``<port>`` values for the correlator's *product controller*
  is yielded after startup.

In both cases, a user can connect to ``<host>:<port>`` and issue a ``?help`` to
see the full range of commands available. The correlator components are
controlled using `katcp`_. Standard katcp requests (such as querying and
subscribing to sensors) are not covered here; only application-specific
requests are listed. Sensors are described in :ref:`monitoring-sensors`.

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

End-to-end correlator shutdown
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A user can issue a ``?product-deconfigure`` command to the correlator's
product controller by connecting to its ``<host>:<port>`` via a line-based
networking utility. This command triggers the stop procedure of all engines
and dsims running in the target correlator. More specifically:

* the product controller instructs the orchestration software to stop the
  containers running the engines,
* which is received by the engines as a ``SIGTERM``,
* finally triggering a ``halt`` in the engines for a graceful shutdown.

As discussed in their standalone documents, the shutdown procedures are vastly
similar between the dsim, fgpu and xbgpu. Ultimately they all:

* finish calculations on data currently in their pipelines,
* stop the transmission of their SPEAD descriptors, and
* in the case of ``fgpu`` and ``xbgpu``, stop their ``spead2`` receivers.

  *  This allows for a more natural engine of internal processing operations.

Individual engine shutdown
^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you've sufficiently tested, debugged and/or reached the desired level of
confusion, there are two options for engine shutdown:

#. simply issue a ``Ctrl + C`` in the terminal window where the engine was
   invoked, or
#. connect to the engine's ``<host>:<port>`` via a line-based networking utility
   and issue a ``?halt``.

  *  As mentioned in the :ref:`indiv-engine-startup`, the host for a
     hand-cranked engine is likely the machine it is run on (i.e.
     ``localhost``), so
  *  The ``port`` parameter is of most importance here.

After either of these approaches are executed, the engine will shutdown cleanly
and quietly according to the stop procedure discussed in its dedicated
document.

Of course, the :program:`fsim` just requires a ``Ctrl + C`` to end operations -
no ``katcp`` commands supported here.

A fair bit of work has gone into ensuring the engines and
:external+aiokatcp:py:class:`DeviceServers <aiokatcp.server.DeviceServer>`
they're built on are robust to a variety of exceptions and anomalies. Adding to
that, the reporting of errors and exceptions has been consolidated for ease of
traceability, e.g. according to each stage of the processing chain (receive,
gpu-processing, transmit). This reduces the potential chaos involved in
monitoring correlator-wide operations.
