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
additional antennas and longer baselines. This package (``katgpucbf``) is
intended for deployment with MeerKAT Extension.

.. _SARAO: https://www.sarao.ac.za/about/sarao/
.. _MeerKAT: https://www.sarao.ac.za/science/meerkat/about-meerkat/
.. _SKA: https://www.skao.int/en/about-us/skao


Radio Astronomy Correlators
---------------------------

Radio astronomy interferometry is a means of achieving higher sensitivity and
resolution in radio telescopes by combining signals from multiple antennas,
as opposed to relying on ever-larger individual antennas which are expensive
and unwieldy.

This combination of signals is typically achieved digitally by equipment known
as a correlator. In the narrowband case, correlation is achieved by cross-
multiplying each antenna's signal with each other antenna's signal. Since modern
radio telescopes have wideband receivers, the signal is decomposed (typically
using the Fourier transform) into multiple narrowband frequency channels.

Mathematically, correlation and frequency decomposition can be done in any
order. A correlator which calculates cross-correlations first and then frequency
decomposition is referred to as X-F, while vice-versa is known as F-X. For
practical reasons (both in terms of compute and interconnect), F-X correlators
are more cost-effective to implement, and so MeerKAT and MeerKAT Extension make
use of this architecture for their correlators.

MeerKAT and MeerKAT Extension's correlators were developed with the influence
of `CASPER`_. Costs are minimised by using commercially-available products
wherever possible. In particular, Ethernet is used to connect signal-processing
nodes. This eliminates the need to design costly custom backplanes.

.. _CASPER: https://casper.berkeley.edu/


This package
------------

This package (``katgpucbf``) provides a software implementation of the DSP
engines of a radio astronomy correlator described above.

The package contains several executable entry-points. The main functionality is
implemented in :program:`fgpu`, :program:`xbgpu` and :program:`vgpu` which
execute (respectively) an F-, XB-, or V-engine. The F-engine implements the
channelisation component of the correlator's operation, while the XB-engine
calculates the correlation products (X) and beamformer output (B). The
V-engine is a tool to resample beamformer output and encode it in a `VDIF`_
format that makes a beam suitable for use as a station in Very Long Baseline
Interferometry (VLBI). The V-engine is still being developed and is not yet
functional.

Additionally, packet simulators are provided for testing purposes. A :ref:`dsim`
can be used to test either an F-engine or an entire correlator. An
:ref:`feng-packet-sim` can be used to test an XB-engine in isolation.

The package also includes unit tests (:file:`test/`), as well as a framework for
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

.. _VDIF: https://vlbi.org/vlbi-standards/vdif/


Controller
----------

.. todo::  ``NGC-680``
    - Relationship with katsdpcontroller
    - reference to a later section which will describe it more thoroughly.
