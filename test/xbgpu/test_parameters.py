"""Module containing correlator configuration options that should be tested where appropriate in all unit tests."""

# These are the estimated subarray sizes that will be run. The 130, 192 and 256 values are estimates for SKA and are
# not final. Additionally values 5,23,61 and 19 are just there to test that various non-power-of-two array sizes will
# run.
array_size = [4, 8, 16, 32, 64, 84, 130, 192, 256, 5, 23, 61, 19]
