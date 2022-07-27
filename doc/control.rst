Operation
=========

katsdpcontroller
----------------
This package (katgpucbf) provides the components of a correlator (engines and simulators),
but not the mechanisms to start up and orchestrate all the components as a
cohesive unit. That is provided by `katsdpcontroller`_.

.. _katsdpcontroller: https://github.com/ska-sa/katsdpcontroller

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
only application-specific requests are listed.

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

.. todo::

   Link to sensor descriptions once they're written in the monitoring section.


Shutting down the correlator
----------------------------

.. todo::  ``NGC-686``
    Describe how to shut the correlator down. Product or master controller
    passes requests on to individual running instances.
