This directory contains tests that were run towards the end of March 2022, for
initial, first-pass verification that the correlator as a whole does what we
expect it to do.

The following tests are included:

- Channelisation or channel response shape,
- Baselines, checking whether signal comes where we expect it to, and
- Delays, checking whether delay and phase tracking are actually doing something.

In each case, we are simply eyeballing the output, and no numerical verification of
whether things are meeting spec is attempted at this stage. This is beyond the
scope of a few scripts in a scratch directory.

The following command was used to start the correlator:
```
sim_correlator.py \
    --name jsmith \
    -a 4 \
    -c 8192 \  # Note: we focus on an 8k correlator for MK+ requirements.
    --band l \
    -d 1 \
    --image-override katgpucbf:harbor.sdp.kat.ac.za/cbf/katgpucbf:main-20220328 \
    lab5.sdp.kat.ac.za
```
