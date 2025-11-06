Monitoring
==========
There are two production mechanisms for monitoring components of the
correlator: `katcp`_ sensors and `Prometheus`_ metrics.

These have different strengths and weaknesses. Prometheus is highly scalable
(so can handle a large number of metrics), supports metric labels, and has a
rich ecosystem (such as `Grafana`_ for easily visualising metrics). On the
other hand, katcp sensors can be more precisely timestamped and support
arbitrary string values (often containing structured data) rather than just
floating-point.

In general, we use katcp sensors for information that should be archived
alongside the data to facilitate its interpretation, as well as sensors that
are needed by other subsystems in the MeerKAT telescope. Prometheus metrics
are used for detailed health and performance monitoring. However, this rule is
not hard-and-fast, and some information is reported via sensors for historical
reasons.

.. _katcp: https://katcp-python.readthedocs.io/en/latest/_downloads/361189acb383a294be20d6c10c257cb4/NRF-KAT7-6.0-IFCE-002-Rev5-1.pdf
.. _Prometheus: https://prometheus.io/
.. _Grafana: https://grafana.com

There is a third monitoring mechanism (event monitoring) intended for
development purposes.

.. _monitoring-sensors:

katcp sensors
-------------
The katcp sensors are self-documenting: issuing a ``?sensor-list`` request to
any of the servers will return a list of the sensors with descriptions. We
thus limit this section to sensors that need a more detailed explanation.

It should be noted that a large number of sensors describing static
configuration (number of channels, accumulation length and so on) are provided
by the product controller rather than this module.

``steady-state-timestamp``
    This sensor is provided by ``dsim``, ``fgpu`` and ``xbgpu``. It can be used to
    synchronise katcp requests with the data. After issuing a katcp request
    that will alter the data stream (such as ``?signals``, ``?gain`` or
    ``?delay``), query the sensor. It will contain an ADC timestamp. Any data
    received with that timestamp or greater will be up to date with the
    effects of all prior requests.

    It should be noted that a ``?delay`` request with a future load time is
    considered to have taken effect when the delay model has been updated,
    even if that load time has not yet been reached.

    It should also be noted that the sensor value from a ``dsim`` does not
    take into account any delays applied by F-engines. One should add the
    delay of the corresponding F engine (or an upper bound on it) to obtain a
    safe timestamp for post–F-engine data streams.

``signals``, ``period`` and ``dither-seed`` (dsim)
    To reproduce the output of the dsim exactly, it is necessary to save all
    three of these sensors, and pass the ``dither-seed`` back to the
    :option:`!--dither-seed` command-line option and the other two to the
    ``?signals`` request (as well as keeping other command-line arguments the
    same).

Prometheus metrics
------------------
To enable Prometheus metrics for any of the services, pass
:option:`!--prometheus-port` to specify the port number. The metrics are then
made available at the ``/metrics`` HTTP endpoint on that port. Pointing a web
browser at that endpoint will show the available metrics with their
documentation.

Event monitoring
----------------
It is also possible to perform very detailed monitoring of events occurring
within fgpu and xbgpu, particularly related to time spent waiting on queues.
This is intended for debugging performance issues rather than production use,
as it has much higher overhead than the other monitoring mechanisms. To
activate it, pass :option:`!--monitor-log` with a filename to the process. It
will write a file with a JSON record per line. The helper script in
:program:`scratch/plot.py` can be used to show a visualisation of the various
queues over time. It's not recommended for more than a few seconds of data.
