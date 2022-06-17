Controlling the Correlator
==========================

katsdpcontroller
----------------

.. todo::

    Describe katsdpcontroller, its role, note that the module can be used
    without it and whatever is used in its place will need to implement the
    functionality described in this "chapter".

    Important to note is that we try to make interacting with katsdpcontroller
    as similar as possible compared to interacting with the individual engines,
    for ease of understanding.


Starting the correlator
-----------------------

.. todo::

    Describe how a correlator should be started. Master controller figures out
    based on a set of input parameters, how to invoke a few instances of
    katgpucbf as dsim, fgpu or xbgpu.


Controlling the correlator
--------------------------

.. todo::

    Describe how the correlator is controlled. This will mostly be delays and
    gains. Product controller passes almost identical requests on to relevant
    instances of katgpucbf.


Shutting down the correlator
----------------------------

.. todo::

    Describe how to shut the correlator down. Product or master controller pass
    pass requests on to individual running instances.

Monitoring
----------

.. todo::

    - Describe KATCP sensors.
    - Describe Prometheus monitoring capabilities.
    - Probably also a good idea to mention the general logic distinguishing
      between what goes to katcp and what to prometheus.
