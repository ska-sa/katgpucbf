Qualification framework
=======================
While the unit tests ensure that individual pieces of the CBF work
correctly, the qualification tests ensure that the system as a whole functions
correctly and meets requirements. In software engineering terms these are
integration tests.

The qualification tests are stored in the :file:`qualification` directory
and are run with pytest. In addition to the usual pass or fail indication, the
tests produce a report (in PDF format), which describes which tests were run,
the steps involved in the tests, the machines used and so on. It also includes
assorted plots showing results.

The tests do not run the katgpucbf code from the local machine. Instead, they
connect to an `SDP Master Controller`_ and use it to start appropriate
CBFs which they interact with. Facilities are provided for the test to
interact with the CBF, both by sending it KATCP requests and by
ingesting the output data. It's thus necessary to have a master controller set
up (which is beyond the scope of this document) and to have a Docker image of
katgpucbf stored in a Docker registry.

.. _SDP Master Controller: https://github.com/ska-sa/katsdpcontroller

Additionally, the hosts in the cluster must be monitored by Prometheus, so that
the qualification report can include information on the hardware and software
configuration. They must run `node-exporter`_ with the arguments
``--collector.cpu.info`` and ``--collector.ethtool``.

.. _node-exporter: https://github.com/prometheus/node_exporter

Requirements
------------

A :file:`requirements.in` and :file:`requirements.txt` are provided in this
directory, based on katgpucbf's :file:`requirements-dev.txt`. A
pared-down version of this may become available in future.

It's necessary to have ``katgpucbf`` installed for the qualification tests to
run, but it is not necessary to have a GPU or CUDA installed. The necessary
parts can be installed with

.. code:: sh

   pip install ".[qualification]"

The machine running the tests needs to be able to receive data from the
CBF network. The data rate can become quite high for larger array sizes.

Configuration
-------------
You will need to create a :file:`qualification/pytest.ini` file.
It is specific to your test environment, so do not commit it to
git. You'll need to set it up only once per machine that you're deploying on,
and it'll look something like this:

.. code:: ini

   [pytest]
   tester = Your Name
   asyncio_mode = auto
   master_controller_host = lab5.sdp.kat.ac.za
   master_controller_port = 5001
   prometheus_url = http://lab5.sdp.kat.ac.za:9090
   product_name = bobs_qualification_cbf  # Use your own name
   interface = enp193s0f0
   use_ibv = true
   log_cli = true
   log_cli_level = info
   addopts = --report-log=report.json

Only set ``use_ibv`` if the NIC and the system support ibverbs. See the
spead2 :external+spead2:doc:`documentation <py-ibverbs>` for advice on setting
that up. This will probably be needed to successfully test large numbers of
channels or antennas.

Running
-------

Use the following command to run the tests contained in
this directory:

.. code:: sh

   spead2_net_raw pytest -v qualification --image-override katgpucbf:harbor.sdp.kat.ac.za/cbf/katgpucbf:latest

Explanation:

-  ``spead2_net_raw`` enables ibverbs usage (see ``use_ibv`` above)
-  ``--image-override`` is designed to work in exactly the same way as
   that in ``sim_correlator.py``, specifying exactly which Docker image
   to use for the tests.

The general pytest options apply, so for instance with ``-x`` you can
stop after the first failed test instead of continuing, etc.

Post-processing
---------------

The steps above produce a ``report.json`` file. To turn that into a usable PDF,
run

.. code:: sh

   qualification/report/generate_pdf.py report.json report.pdf

This requires at least ``texlive-base``, ``texlive-latex-extra``, ``texlive-science`` and
``latexmk``. This step doesn't interact with the live system at all, so it is
possible to copy/mount the JSON file to another machine to run this step.
