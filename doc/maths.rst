Mathematical background
=======================

This section is not intended as a tutorial for radio astronomy. It briefly
summarises some key equations, with the intent of illustrating implementation
choices made by this package.

Frequencies
-----------
This package works only in positive baseband frequencies, and is unaware of
heterodyne systems. Where necessary, the signal will need to be mapped to this
range before being provided as input. As an example, MeerKAT L-band digitisers
receive signal in the range 856–1712 MHz, but mix it down to 0–856 MHz by
negating every second sample.

Complex voltages
----------------
A wave with frequency :math:`\omega` and wave number :math:`k` is considered
to have a phasor of

.. math::

   e^{(\omega t - kz)j}

where :math:`t` is time and :math:`z` is position. In particular, phase
measured at a fixed position (an antenna) increases with time.

Polyphase filter bank
---------------------
A finite impulse response (FIR) filter is applied to the signal to condition
the frequency-domain response. The filter is the product of a Hann window (to
reduce spectral leakage) and a sinc (to broaden the peak to cover the
frequency bin). Specifically, if there are :math:`n` output channels and
:math:`t` taps in the polyphase filter bank, then the filter has length
:math:`w = 2nt`, with coefficients

.. math::

   \DeclareMathOperator{\sinc}{sinc}
   x_i = A\sin^2\left(\frac{\pi i}{w - 1}\right)
         \sinc\left(\frac{i + \tfrac 12 - nt}{2n}\right),

where :math:`i` runs from 0 to :math:`w - 1`. Here :math:`A` is a
normalisation factor which is chosen such that :math:`\sum_i x_i^2 = 1`. This
ensures that given white Gaussian noise as input, the expected output power
in a channel is the same as the expected input power in a digitised sample.
Note that the input and output are treated as integers rather than as
fixed-point values.

Correlation products
--------------------
Given a baseline (p, q) and time-varying channelised voltages :math:`e_p` and
:math:`e_q`, the correlation product is the sum of :math:`e_p \overline{e_q}`
over the accumulation period. This is computed in integer arithmetic and so is
lossless except when saturation occurs.

Narrowband
----------
.. todo:: Document the down-conversion filter

Delay and phase compensation
----------------------------
.. todo:: Document delay and phase compensation
