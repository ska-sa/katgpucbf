# Katgpucbf prototype qualification framework

The software in this folder is intended for use with pytest in order to run
qualification tests on a correlator run using `katgpucbf`.

A connection is made to an SDP Master Controller, an appropriately-specced
correlator product is requested, interacted with, and then shut down afterwards.
Facilities are provided for the test to interact with the correlator, both by
sending it KATCP requests and by ingesting the output data.

## Requirements

A `requirements.in` and `requirements.txt` is provided in this directory, based
on `katgpucbf`'s `requirements-dev.txt`. A pared-down version of this may become
available in future.

While `katgpucbf` itself doesn't need to be installed for the test framework to
work, you need to have the appropriate Docker image available as specified (see
below).

## Sample pytest.ini

I don't want to commit a `pytest.ini` for this folder to git, but it's helpful
to have one. You'll need to set it up only once per machine that you're
deploying on, and it'll look something like this:

```
[pytest]
asyncio_mode = auto
master_controller_host = lab5.sdp.kat.ac.za
master_controller_port = 5001
interface = enp193s0f0
use_ibv = true
log_cli = true
log_cli_level = info
addopts = --report-log=report.json
```

## Running

Currently, I use the following command to run the tests contained in this
directory:

```
spead2_net_raw pytest -v  qualification --image-override katgpucbf:harbor.sdp.kat.ac.za/cbf/katgpucbf:main-20220328
```

Explanation:

* `spead2_net_raw` enables ibverbs usage. For a small correlator, this shouldn't
  really be an issue because the output rates are quite low, but for larger ones
  it will be an issue.
* `--image-override` is designed to work in exactly the same way as that in
  `sim_correlator.py`, specifying exactly which Docker image to use for the
  tests. This is just the example that I've been using, update accordingly.

The general pytest options apply, so for instance with `-x` you can stop after
the first failed test instead of continuing, etc.

## Post-processing

I've been using the same `generate_pdf_test_report` currently living in
`dc_sand/test_report` - so copy the `report.json` output from the test run to
a local machine, and this should be all you need to generate a PDF report for
your general reading pleasure.
