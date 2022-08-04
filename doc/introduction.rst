Introduction
============


MeerKAT and MeerKAT Extension
-----------------------------

.. todo:: ``NGC-667``
    - What SARAO is
    - What MeerKAT and MK+ are
    - Maybe some links


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
an F- or an XB-engine.

Additionally, packet simulators are provided for testing purposes. A digitiser
simulator (:program:`dsim`) can be used to test either an F-engine or an entire
correlator. An F-engine simulator (:program:`fsim`) can be used to test an
XB-engine in isolation.

The module also includes unit tests (``test/``), as well as a framework for
automated testing of an entire correlator against the set of requirements
applicable to the MeerKAT Extension CBF (``qualification/``).


Controller
----------

.. todo::  ``NGC-680``
    - Relationship with katsdpcontroller
    - reference to a later section which will describe it more thoroughly.
