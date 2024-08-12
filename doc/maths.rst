Mathematical background
=======================

This section is not intended as a tutorial for radio astronomy. It briefly
summarises some key equations, with the intent of illustrating implementation
choices made by this package.

Frequencies
-----------
This package works only in positive baseband frequencies, and is unaware of
heterodyne systems. Where necessary, the signal will need to be mixed to this
range before being provided as input. As an example, MeerKAT L-band digitisers
receive signal in the range 856–1712 MHz, but mix it down to 0–856 MHz by
negating every second sample (the digital equivalent of a 856 MHz mixing
signal).

This has implications for :ref:`delay compensation <math-delay>`.

Complex voltages
----------------
A wave with frequency :math:`f` and wave number :math:`k` is considered
to have a phasor of

.. math::

   e^{(2\pi ft - kz)j}

where :math:`t` is time and :math:`z` is position. In particular, phase
measured at a fixed position (an antenna) increases with time.

Correlation products
--------------------
Given a baseline (p, q) and time-varying channelised voltages :math:`e_p` and
:math:`e_q`, the correlation product is the sum of :math:`e_p \overline{e_q}`
over the accumulation period. This is computed in integer arithmetic and so is
lossless except when saturation occurs.

Narrowband
----------
.. todo:: Document the down-conversion filter

.. _math-delay:

Delay and phase compensation
----------------------------
The delay sign convention is such that a input voltage sample with timestamp
:math:`t` will have an output timestamp of :math:`t + d` (where :math:`d` is
the delay). In other words, the specified values are amounts by which the
incoming signals should be delayed to align them.

To correctly apply delay with sub-sample precision, it is necessary to know
the original ("sky") frequency of the signal, before mixing and aliasing to
baseband. The user supplies this indirectly by specifying the phase correction
that must be applied at the centre frequency, i.e. :math:`-2\pi f_c d`, where
:math:`f_c` is the centre frequency. This calculation is provided by
:class:`katpoint.delay.DelayCorrection`.
