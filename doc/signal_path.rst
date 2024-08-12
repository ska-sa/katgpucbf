Signal path overview
====================

This section gives a logical overview of the signal path. The actual
implementation may split or combine steps for efficiency; refer to the
:doc:`fgpu.design` and :doc:`xbgpu.design` sections for details.

Edges in diagrams are annotated to indicate the data type. The following types
are used:

:samp:`i{N}`
  Signed integer or fixed-point with :samp:`{N}` total bits
:samp:`u{N}`
  Unsigned integer or fixed-point with :samp:`{N}` bits
:samp:`f{N}`
  Floating-point with :samp:`{N}` bits, following IEEE 754-2019.
:samp:`c{X}`
  Complex values formed from real and imaginary components of type :samp:`{X}`
  e.g., ``cf32`` is single-precision floating point complex.

Channelisation
--------------
The figure below shows the signal path for the wide-band case. The dotted
boxes and arrows represent control parameters that can be adjusted at
runtime.

Note that the input and output bit depths (shown as ``i10`` and ``ci8`` on the
diagram) are configurable. Between unpacking and quantisation, all
calculations are performed in single precision. Since the input has a bounded
range, overflow is only possible at the quantisation step (which saturates).

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
   \node[control, below=of delays] (eq) {Eq coefficients};
   \draw[->] (receive)
     -| node[lbl, very near start, auto, swap] {V}
        node[lbl, near end, auto, swap] {i10} (unpackx);
   \draw[->] (receive)
     -| node[lbl, very near start, auto] {H}
        node[lbl, near end, auto] {i10} (unpacky);
   \draw[->, dotted] (delays) to[lbl, auto, swap, edge label=i32] (cdelayx);
   \draw[->, dotted] (delays) to[lbl, auto, swap, edge label=f32] (eqx);
   \draw[->, dotted] (delays) to[lbl, auto, edge label=i32] (cdelayy);
   \draw[->, dotted] (delays) to[lbl, auto, edge label=f32] (eqy);
   \draw[->, dotted] (eq) to[lbl, auto, swap, edge label=cf32] (eqx);
   \draw[->, dotted] (eq) to[lbl, auto, edge label=cf32] (eqy);
   \draw[->] (quantx) |- node[lbl, auto, swap, near start] {ci8} (pack);
   \draw[->] (quanty) |- node[lbl, auto, near start] {ci8} (pack);

Polyphase filter bank
^^^^^^^^^^^^^^^^^^^^^
A finite impulse response (FIR) filter is applied to the signal to condition
the frequency-domain response. The filter is the product of a Hann window (to
reduce spectral leakage) and a sinc (to broaden the peak to cover the
frequency bin). Specifically, if there are :math:`n` output channels and
:math:`t` taps in the polyphase filter bank (PFB), then the filter has length
:math:`w = 2nt`, with coefficients

.. math::

   x_i = A\sin^2\left(\frac{\pi i}{w - 1}\right)
         \operatorname{sinc}\left(w_c\cdot \frac{i + \tfrac 12 - nt}{2n}\right),

where :math:`i` runs from 0 to :math:`w - 1`. Here :math:`A` is a
normalisation factor which is chosen such that :math:`\sum_i x_i^2 = 1`. This
ensures that given white Gaussian noise as input, the expected output power
in a channel is the same as the expected input power in a digitised sample.
Note that the input and output are treated as integers rather than as
fixed-point values.

The tuning parameter :math:`w_c` (specified by the :option:`!--w-cutoff`
command-line option) scales the width of the response in the frequency domain.
The default value is 1, which makes the width of the response (at -6dB)
approximately equal the channel spacing.
