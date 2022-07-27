Operation
=========

.. todo::

    If this section gets too much, it can possibly make its way into its own
    ``controlling.rst`` file or some such.

katsdpcontroller
----------------

.. todo::  ``NGC-683``
    Describe katsdpcontroller, its role, note that the module can be used
    without it and whatever is used in its place will need to implement the
    functionality described in this "chapter".

    Important to note is that we try to make interacting with katsdpcontroller
    as similar as possible compared to interacting with the individual engines,
    for ease of understanding.


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

There are two main scenarios which involve the shutting down of a correlator
and its constituent engines.

#. During normal correlator operation, and
#. During testing and debugging of individual engines and/or dsims.

Normal correlator operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
As previously mentioned, currently :mod:`katgpucbf`'s correlator-wide
orchestration is done via `katsdpcontroller`_. This, in turn, provides an
interface to the correlator and its constituent engines based on an
:external+aiokatcp:doc:`aiokatcp server <server/tutorial>`. For this reason, a
user can connect to the correlator's ``<ip_addr>:<port>`` using a networking
utility like ``netcat`` (`nc`_) and issue a ``?product-deconfigure`` command.

.. note::
    A sidebar to plug a utility written by one of :mod:`katgpucbf`'s
    developers. `ntsh`_ makes this line-based protocol interaction much easier
    to follow for beginner (and more experienced) users.

This ``?product-deconfigure`` command triggers the stop procedure of all
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

.. _katsdpcontroller: https://github.com/ska-sa/katsdpcontroller
.. _nc: https://www.commandlinux.com/man-page/man1/nc.1.html
.. _ntsh: https://pypi.org/project/ntsh/

Running individual Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^
An example of this scenario is running a standalone instance of ``xbgpu`` -
along with an appropriately-configured ``fsim``.

* Here, you might use one of the handy scripts under e.g. ``scratch/xbgpu/``
  to launch an XB-Engine instance.
* Once you've sufficiently debugged and/or reached the desired level of
  confusion, you can simply issue a ``Ctrl + C`` in your terminal window.
* ``xbgpu`` will shut down cleanly and quietly according to the stop procedure
  mentioned above.

A fair bit of work has gone into ensuring the engines and
:external+aiokatcp:py:class:`DeviceServers <aiokatcp.server.DeviceServer>`
they're built on are robust to a variety of exceptions and anomalies. Adding to
that, the reporting of errors and exceptions has been consolidated for ease of
traceability, e.g. according to each stage of the processing chain (receive,
gpu-processing, transmit). This reduces the potential chaos involved in
monitoring correlator-wide operations.
