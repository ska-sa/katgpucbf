Introduction
============


MeerKAT and MeerKAT Extension
-----------------------------

The South African Radio Astronomy Observatory (`SARAO`_) manages all radio
astronomy initiatives and facilities in South Africa, including the `MeerKAT`_
radio telescope. MeerKAT is a precursor to the Square Kilometre Array (`SKA`_)
and consists of 64 offset-Gregorian antennas in the Karoo desert in South
Africa.

MeerKAT Extension is a project currently underway to extend MeerKAT with
additional antennas and longer baselines. This module (``katgpucbf``) is
intended for deployment with MeerKAT Extension.

.. _SARAO: https://www.sarao.ac.za/about/sarao/
.. _MeerKAT: https://www.sarao.ac.za/science/meerkat/about-meerkat/
.. _SKA: https://www.skao.int/en/about-us/skao


Radio Astronomy Correlators
---------------------------

.. todo::  ``NGC-678``
    - Correlators are for correlating
    - F-X architecture
    - Ethernet interconnect



This module
-----------

This module (``katgpucbf``) provides a software implementation of the DSP
engines of a radio astronomy correlator described above.

The module contains several executable entry-points. The main functionality is
implemented in :program:`fgpu` and :program:`xbgpu` which execute (respectively)
an F- or an XB-engine. The F-engine implements the channelisation component of
the correlator's operation, while the XB-engine calculates the correlation
products (X) and beamformer output (B). The beamformer component is not
currently implemented, but is planned for a future release.

Additionally, packet simulators are provided for testing purposes. A :ref:`dsim`
can be used to test either an F-engine or an entire correlator. An
:ref:`feng-packet-sim` can be used to test an XB-engine in isolation.

The module also includes unit tests (:file:`test/`), as well as a framework for
automated testing of an entire correlator against the set of requirements
applicable to the MeerKAT Extension CBF (:file:`qualification/`).

As far as possible, the code in this package is not MeerKAT-specific. It could
in theory be used at other facilities, provided that compatible input and output
formats are used (including number of input and output bits). The
:mod:`katgpucbf.meerkat` module contains some tables that are specific to
MeerKAT and MeerKAT Extension, which are used by some convenience scripts, but
which are not used by the core programs.

Some additional scripts (:file:`scratch/`) which the developers have found to be
useful are included, but user discretion is advised as these aren't subject to
very much quality control, and will need to be adapted to your environment.


Controller
----------

.. todo::  ``NGC-680``
    - Relationship with katsdpcontroller
    - reference to a later section which will describe it more thoroughly.
