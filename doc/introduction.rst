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

Some additional scripts (:file:`scratch/`) which the developers have found to be
useful are included, but user discretion is advised as these aren't subject to
very much quality control, and will need to be adapted to your environment.


Controller
----------

.. todo::  ``NGC-680``
    - Relationship with katsdpcontroller
    - reference to a later section which will describe it more thoroughly.
