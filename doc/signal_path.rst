Signal path overview
====================

This section gives a logical overview of the signal path. The actual
implementation is mathematically equivalent (when ignoring floating-point
rounding errors), but it splits, combines, and reorders steps for efficiency;
refer to the :doc:`fgpu.design` and :doc:`xbgpu.design` sections for details.

Edges in diagrams are annotated to indicate the data type. The following types
are used:

:samp:`i{N}`
  Signed integer or fixed-point with :samp:`{N}` total bits
:samp:`f{N}`
  Floating-point with :samp:`{N}` bits, following IEEE 754-2019.
:samp:`c{X}`
  Complex values formed from real and imaginary components of type :samp:`{X}`
  e.g., ``cf32`` is single-precision floating point complex.

Dotted boxes and arrows represent control parameters that can be adjusted at
runtime.

Channelisation and delay correction
-----------------------------------
Note that the input and output bit depths (shown as ``i10`` and ``ci8`` on the
diagram) are configurable. Between unpacking and quantisation, all
calculations are performed in single precision. Since the input has a bounded
range, overflow is only possible at the quantisation step (which saturates).

The figure below shows the signal path for wide-band channelisation.

.. tikz:: Signal path for wide-band channelisation.
   :libs: chains, positioning

   \tikzset{
     base/.style={minimum width=2.5cm, minimum height=1cm, align=center},
     op/.style={draw, base},
     control/.style={draw, base, rounded corners, dotted},
     lbl/.style={font=\scriptsize},
     every join/.style={draw,->},
     >=latex,
   }
   \newcommand{\side}[2]{
     \node[op, on chain, join=by {#2, edge label=f32}] (cdelay#1) {Coarse delay};
     \node[op, on chain, join=by {#2, edge label=f32}] (pfb#1) {PFB};
     \node[op, on chain, join=by {#2, edge label=cf32}] (eq#1) {Fine delay\\ Equalisation};
     \node[op, on chain, join=by {#2, edge label=cf32}] (dither#1) {Dither};
     \node[op, on chain, join=by {#2, edge label=cf32}] (quant#1) {Quantise};
   }
   \node[op] (receive) {Receive};
   \begin{scope}[start chain=chainx going below]
     \node[op, below left=of receive, on chain] (unpackx) {Unpack};
     \side{x}{lbl, swap}
   \end{scope}
   \begin{scope}[start chain=chainy going below]
     \node[op, below right=of receive, on chain] (unpacky) {Unpack};
     \side{y}{lbl}
   \end{scope}
   \begin{scope}[start chain=sink going below]
     \node[op, on chain, below right=of quantx] (pack) {Corner turn\\ Pack};
     \node[op, on chain, join=by {lbl, edge label=ci8}] (transmit) {Transmit};
   \end{scope}
   \node[control, right=of pfbx] (delays) {Delays};
   \node[control, right=of eqx] (eq) {Eq coefficients};
   \draw[->] (receive)
     -| node[lbl, very near start, auto, swap] {pol0}
        node[lbl, near end, auto, swap] {i10} (unpackx);
   \draw[->] (receive)
     -| node[lbl, very near start, auto] {pol1}
        node[lbl, near end, auto] {i10} (unpacky);
   \draw[->, dotted] (delays) to[lbl, auto, edge label'=i32] (cdelayx);
   \draw[->, dotted] (delays) to[lbl, auto, edge label=f32] (eqx);
   \draw[->, dotted] (delays) to[lbl, auto, edge label=i32] (cdelayy);
   \draw[->, dotted] (delays) to[lbl, auto, edge label'=f32] (eqy);
   \draw[->, dotted] (eq) to[lbl, auto, edge label'=cf32] (eqx);
   \draw[->, dotted] (eq) to[lbl, auto, edge label=cf32] (eqy);
   \draw[->] (quantx) |- node[lbl, auto, swap, near start] {ci8} (pack);
   \draw[->] (quanty) |- node[lbl, auto, near start] {ci8} (pack);

Delay
^^^^^
Delays may be specified with sub-sample precision. To handle this, the delay
is split into two components: a :dfn:`coarse` delay (a whole number of
samples) and a :dfn:`fine` delay (between -0.5 and +0.5 samples). The coarse
delay is applied as a shift in time, while the fine delay is applied as a
phase slope in the frequency domain. As noted in :ref:`math-delay`, the user
provides the overall phase adjustment for the centre frequency, and the
constant term of the phase slope is computed from that (taking into account
the effect of the coarse delay on phase).

The fine delay and the fixed phase offset for each spectrum are computed in
double precision then reduced to single precision for application. Conversion
of the delay to a per-channel phase correction, and of phases to complex
phasors are done in single precision.

Polyphase filter bank (PFB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
A finite impulse response (FIR) filter is applied to the signal to condition
the frequency-domain response. The filter is the product of a window function
(to reduce spectral leakage) and a sinc (to broaden the peak to
cover the frequency bin). Specifically, if there are :math:`n` output channels
and :math:`t` taps in the polyphase filter bank, then the filter has length
:math:`w = 2nt`, with coefficients

.. math::

   x_i = AW_i\operatorname{sinc}\left(w_c\cdot \frac{i + \tfrac 12 - nt}{2n}\right),

where :math:`i` runs from 0 to :math:`w - 1`, and :math:`W` is the window function,
for which there are two choices:

- Hann: :math:`W_i = \sin^2\left(\frac{\pi i}{w - 1}\right)`
- Rect: :math:`W_i = 1`.

:math:`A` is a normalisation factor which is chosen such that :math:`\sum_i
x_i^2 = 1`. This ensures that given white Gaussian noise as input, the
expected output power in a channel is the same as the expected input power in
a digitised sample. Note that the input and output are treated as integers
rather than as fixed-point values.

The tuning parameter :math:`w_c` (specified by the :option:`!--w-cutoff`
command-line option) scales the width of the response in the frequency domain.
The default value is 1, which makes the width of the response (at -6dB)
approximately equal the channel spacing.

In some cases spectral leakage is less important than the ability to
reconstruct the original signal. Setting :math:`t = 1`, :math:`w_c = 0` and
using the rectangular window function gives a degenerate PFB in which each
block of :math:`2n` samples is Fourier transformed.

.. _signal-path.narrow:

Dithering
^^^^^^^^^
To improve linearity, a random value selected uniformly from the interval
(-0.5, 0.5) is added to each component (real and imaginary) before
quantisation. The random seeds are carefully chosen to ensure that
random sequences are not shared across antennas.

Narrowband
^^^^^^^^^^
Narrowband outputs are those in which only a portion of the digitised
bandwidth is channelised and output. Typically they have narrower channel
widths. The overall approach is as follows:

1. The signal is multiplied (:dfn:`mixed`) by a complex tone of the form
   :math:`e^{2\pi jft}`, to effect a shift in the frequency of the
   signal. The centre of the desired band is placed at the DC frequency.

2. The signal is convolved with a low-pass filter. This suppresses most
   of the unwanted parts of the band, to the extent possible with a FIR
   filter.

3. The signal is subsampled (every Nth sample is retained), reducing the data
   rate. The low-pass filter above limits aliasing. At this stage, twice as
   much bandwidth as desired is retained. The steps up to this one are
   referred to as :dfn:`digital down-conversion` (DDC).

4. The coarse delay and PFB proceed largely as before, but using double the
   final channel count (since the bandwidth is also doubled, the channel width
   is as desired). The input is now complex rather than real (due to the
   mixing), so the PFB is complex-to-complex rather than real-to-complex.

5. Half the channels (the outer half) are discarded.

.. note::
   To avoid confusion, the "subsampling factor" is the ratio of original to
   retained samples in the subsampling step, while the "decimation factor" is
   the factor by which the bandwidth is reduced. Because the mixing turns a
   real signal into a complex signal, the subsampling factor is twice the
   decimation factor in step 3 (but equal to the overall decimation
   factor).

The decimation is thus achieved by a combination of time-domain (steps 2 and
3) and frequency domain (step 5) techniques. This has better computational
efficiency than a purely frequency-domain approach (which would require the
PFB to be run on the full bandwidth), while mitigating many of the filter
design problems inherent in a purely time-domain approach (the roll-off of the
FIR filter can be hidden in the discarded outer channels).

The figure below shows the modified signal path.

.. tikz:: Signal path for narrow-band channelisation (with new stages in blue).
   :libs: chains, positioning

   \tikzset{
     base/.style={minimum width=2.5cm, minimum height=1cm, align=center},
     op/.style={draw, base},
     extra/.style={draw=blue, color=blue},
     control/.style={draw, base, rounded corners, dotted},
     lbl/.style={font=\scriptsize},
     every join/.style={draw,->},
     >=latex,
   }
   \newcommand{\side}[2]{
     \node[op, extra, on chain, join=by {#2, edge label=cf32}] (ddc) {DDC};
     \node[op, on chain, join=by {#2, edge label=cf32}] (cdelay#1) {Coarse delay};
     \node[op, on chain, join=by {#2, edge label=cf32}] (pfb#1) {PFB};
     \node[op, extra, on chain, join=by {#2, edge label=cf32}] (discard#1) {Discard\\ channels};
     \node[op, on chain, join=by {#2, edge label=cf32}] (eq#1) {Fine delay\\ Equalisation};
     \node[op, on chain, join=by {#2, edge label=cf32}] (dither#1) {Dither};
     \node[op, on chain, join=by {#2, edge label=cf32}] (quant#1) {Quantise};
   }
   \node[op] (receive) {Receive};
   \begin{scope}[start chain=chainx going below]
     \node[op, below left=of receive, on chain] (unpackx) {Unpack};
     \side{x}{lbl, swap}
   \end{scope}
   \begin{scope}[start chain=chainy going below]
     \node[op, below right=of receive, on chain] (unpacky) {Unpack};
     \side{y}{lbl}
   \end{scope}
   \begin{scope}[start chain=sink going below]
     \node[op, on chain, below right=of quantx] (pack) {Corner turn\\ Pack};
     \node[op, on chain, join=by {lbl, edge label=ci8}] (transmit) {Transmit};
   \end{scope}
   \node[control, right=of pfbx] (delays) {Delays};
   \node[control, right=of eqx] (eq) {Eq coefficients};
   \draw[->] (receive)
     -| node[lbl, very near start, auto, swap] {pol0}
        node[lbl, near end, auto, swap] {i10} (unpackx);
   \draw[->] (receive)
     -| node[lbl, very near start, auto] {pol1}
        node[lbl, near end, auto] {i10} (unpacky);
   \draw[->, dotted] (delays) to[lbl, auto, edge label'=i32] (cdelayx);
   \draw[->, dotted] (delays) to[lbl, auto, edge label=f32] (eqx);
   \draw[->, dotted] (delays) to[lbl, auto, edge label=i32] (cdelayy);
   \draw[->, dotted] (delays) to[lbl, auto, edge label'=f32] (eqy);
   \draw[->, dotted] (eq) to[lbl, auto, edge label'=cf32] (eqx);
   \draw[->, dotted] (eq) to[lbl, auto, edge label=cf32] (eqy);
   \draw[->] (quantx) |- node[lbl, auto, swap, near start] {ci8} (pack);
   \draw[->] (quanty) |- node[lbl, auto, near start] {ci8} (pack);

Discarding half the channels after channelisation allows for a lot of freedom
in the design of the DDC FIR filter: the discarded channels can have an
arbitrary response. This allows for a gradual transition from passband to
stopband. We use :func:`scipy.signal.remez` to produce a filter that is as
close as possible to 1 in the passband and 0 in the stopband. A weighting
factor (which the user can override) balances the priority of the passband
(ripple) and stopband (alias suppression).

The filter performance is slightly improved by noting that the discarded
channels have multiple aliases, and the filter response in those aliases is
also irrelevant. We thus use :func:`scipy.signal.remez` to only optimise the
response to those channels that alias into the output.

Narrowband without discard
~~~~~~~~~~~~~~~~~~~~~~~~~~
The above combined time-frequency approach to narrowband can be disabled,
giving a purely time-domain FIR filter. In this case, step 5 is skipped.
The filter design in this case is more critical, and needs to trade off
factors such as passband ripple, rolloff, and alias rejection.

.. todo::

   Describe the filter design once it is finalised.

The primary use case is for reconstructing a time-domain signal from the
channelised output, where completely discarding channels appears to lose
necessary information.

Correlation
-----------
Given a baseline (p, q) and time-varying channelised voltages :math:`e_p` and
:math:`e_q`, the correlation product is the sum of :math:`e_p \overline{e_q}`
over the accumulation period. This is computed in integer arithmetic and so is
lossless except when saturation occurs.

The figure below shows the signal path.

.. tikz:: Signal path for correlation
   :libs: chains

   \tikzset{
     base/.style={minimum width=2.5cm, minimum height=1cm, align=center},
     op/.style={draw, base},
     control/.style={draw, base, rounded corners, dotted},
     lbl/.style={font=\scriptsize},
     every join/.style={draw,->},
     >=latex,
   }
   \begin{scope}[start chain=going below]
     \node[op, on chain] {Receive};
     \node[op, on chain, join=by {lbl,edge label=ci8}] {Correlate\\ Accumulate};
     \node[op, on chain, join=by {lbl,edge label=ci64}] {Saturate};
     \node[op, on chain, join=by {lbl,edge label=ci32}] {Transmit};
   \end{scope}

Beamforming
-----------
The signal path below is repeated for each single-polarisation beam. Delays
are computed purely with a phase slope in the frequency domain, similarly to
the fine delays in the channeliser. Dithering is done the same way as for
channelisation. Since all calculations are performed in single precision
floating point and the input has a limited range, overflow can only occur
during quantisation (which saturates).

.. tikz:: Signal path for beamforming
   :libs: chains

   \tikzset{
     base/.style={minimum width=2.5cm, minimum height=1cm, align=center},
     op/.style={draw, base},
     control/.style={draw, base, rounded corners, dotted},
     lbl/.style={font=\scriptsize},
     every join/.style={draw,->},
     >=latex,
   }
   \begin{scope}[start chain=going below]
     \node[op, on chain] {Receive};
     \node[op, on chain, join=by {lbl,edge label=ci8}] (mult) {Taper/Scale\\ Delay};
     \node[op, on chain, join=by {lbl,edge label=cf32}] {Sum};
     \node[op, on chain, join=by {lbl,edge label=cf32}] {Dither};
     \node[op, on chain, join=by {lbl,edge label=cf32}] {Quantise};
     \node[op, on chain, join=by {lbl,edge label=ci8}] {Transmit};
     \node[control, above right=of mult] (taper) {Tapering\\ coefficients};
     \node[control, right=of mult] (gain) {Requantisation\\ gain};
     \node[control, below right=of mult] (delay) {Delays};
     \draw[->, dotted] (taper) to[lbl, near start, edge label=f32] (mult);
     \draw[->, dotted] (gain) to[lbl, edge label=f32] (mult);
     \draw[->, dotted] (delay) to[lbl, near start, edge label'=f32] (mult);
   \end{scope}
