# GPU-accelerated F-engine prototype

This is proof-of-concept code, not a fully working F-engine.

## Features

The following features have been implemented, although they are not all fully tested:

- Network receive.
- Decoding of 10-bit digitiser samples.
- Polyphase filter bank. The current filter weights are slightly asymmetric (by
  one sample), which introduces some quirks to the frequency response. Up to 16
  taps works well, after which register pressure impairs performance.
- Delay tracking, with both whole-sample (coarse) and sub-sample (fine) delays,
  although it is essentially untested and there is no interface to enable it.
- Scaling by a single real gain and quantisation to 8-bit output (with
  saturation, but -128..127 rather than -127..127.
- Reordering to the MeerKAT antenna-channelised-voltage heap format, and network
  transmission.

## Missing features

- An interface for real-time control e.g. to adjust gains or set delays etc.
- Fringe rotation (channel-independent, time-varying phase rotation to
  fix up delays after downconversion to baseband).
- Per-channel complex gains.
- Performance monitoring.
- Missing packets are not taken into account properly (the affected output
  heaps need to be suppressed).

## Other TODOs

- Get rid of the print statements and provide a better monitoring interface.
- The frontend processing tends to be run in two pieces on each chunk.
- Use fixed seeds in unit tests.
- More unit tests, particularly for delay model, but also end-to-end testing.
- More generally useful benchmark scripts.
- More control over affinity (for the main Python thread and worker threads).
- Try to prevent CUDA from spinning on waits, to reduce CPU usage.
- Check transmit performance when large number of output groups are used.
- Better testing interfaces e.g. deterministic (single-threaded) pcap
  receive, and some form of lossless output capture.
- Test out large channel counts (256K/512K) for suitability to implement
  narrowband.
- Experiment with doing smaller transfers between host and GPU to better
  exploit L3 cache with DDIO.
