Benchmarking
============
Performing a benchmark of fgpu or xbgpu under realistic conditions (in
particular, with data transmitted over UDP) is difficult because there is no
flow control, and the achievable rate can only be observed indirectly by
running the system at a particular rate and checking whether it keeps up. We
wish to know the largest rate at which we keep up most of the time; we'll call
this the "critical" rate. The :file:`scratch/benchmarks` directory contains a
script to help estimate the critical rate for fgpu (xbgpu support may be added
later).

To run it, you'll need

- two servers (one to generate data and one to receive it)

  - configured for use with :external+spead2:doc:`spead2's ibverbs support
    <py-ibverbs>` on a high-speed network between them;
  - with a Docker daemon;

- somewhere to run the script (referred to below as the "client"), set up for
  key-based SSH access to the servers in accounts that can invoke the Docker
  client.
- a Docker registry containing the katgpucbf Docker image.

The assignment of threads to processor cores on the servers has been
optimised for AMD Epyc servers with at least 16 cores; for other servers or
for lower core counts, the benchmark script may need to be tuned to provide
sensible placement.

The client machine will also need a few Python packages installed; you can
use :command:`pip install -r scratch/benchmarks/requirements.txt` to install
them. You do not need to have katgpucbf installed.

On the client machine, you will need to create a TOML_ file describing the
servers you want to use. Each table in the TOML file describes one server. You
can give them any names you like, but by default the script will use the
servers called ``dsim`` and ``fgpu``. A typical file looks like this:

.. code-block:: toml

   [dsim]
   hostname = 'server01.domain'
   username = 'myusername'
   interfaces = ['enp193s0f0np0', 'enp193s0f1np1']

   [fgpu]
   hostname = 'server02.domain'
   username = 'myusername'
   interfaces = ['enp193s0f0np0', 'enp193s0f1np1']

.. _TOML: https://toml.io/

The `interfaces` arrays list the names of the ibverbs-capable network
interfaces that can be used for sending or receiving the data.

By default the servers are loaded from :file:`servers.toml` in the current
directory, although the :option:`--servers` command-line option can override
this.

With all the requirements in place, change to the :file:`scratch/benchmarks`
directory and run :program:`./benchmark_fgpu.py`. There are some options you
may need:

.. option:: -n <N>

   Set the number of engines to run simultaneously. The benchmark has been
   developed for values 1 and 4, and may need further tuning to effectively
   test other values.

.. option:: --image <image>

   Override the Docker image to run. The default is set up for the lab
   environment at SARAO.

.. option:: --low <rate>, --high <rate>

   Lower and upper bounds on the ADC sample rate. The critical rate will be
   searched between these two bounds. The benchmark will error out if the
   lower bound fails or the upper bound passes.

.. option:: --interval <rate>

   Target confidence interval size. The final result will not be an exact rate
   (because the process is probabilistic) but rather a range. The algorithm
   will keep running until it is almost certain that the critical rate is
   inside an interval of this size or less. Setting this smaller than roughly
   1% of the critical rate can cause the algorithm to fail to converge
   (because there is too much noise to determine the rate more accurately).
   See also :option:`--max-comparisons`.

.. option:: --max-comparisons <N>

   Bound the number of comparisons to perform in the search (excluding the
   sanity checks that the low and high rates are reasonable bounds). If
   :option:`--interval` is too small, the algorithm might not otherwise
   converge. If this number of comparisons is exceeded, a larger interval will
   be output.

.. option:: --step <rate>

   Only multiples of this value will be tested. Some parts of the algorithm
   require time proportional to the square of the number of steps between
   the low and high rate, so this should not be too small. However, it must
   not be larger than :option:`--interval`.

.. option:: --servers <filename>

   Server description TOML file.

.. option:: --dsim-server <name>, --fgpu-server <name>

   Override the server names to find in the servers file.

This is not a complete list of options; run the command with :option:`!--help`
to see others.

Multicast groups
----------------
The benchmark code currently hard-codes a number of multicast groups. Thus,
**two instances cannot be run on the same network at the same time**. The
groups are all in the 239.102.0.0/16 subnet.

Algorithm
---------
The algorithm can be seen as a noisy binary search using Bayesian inference.
Even when running at a slow enough rate, packets may be randomly lost, and
this could send a classical binary search down the wrong side of the search
tree and yield a very incorrect answer.

Instead, the range from :option:`--low` to :option:`--high` is divided into bins of
size :option:`--step`, and for each bin, we keep a probability that the true rate
falls into that bin. We also have a model of how likely a trial is to succeed
at a given rate, if we know the critical rate: very likely/unlikely if the
given rate is significantly lower/higher than the critical rate, and more
uncertain when the given rate is close to critical. After running a trial, we
can use Bayes' Theorem to update the probability distribution. To choose the
binary boundary to test, we consider every option and pick the one that gives
the largest expected decrease in the entropy of the distribution.

Determining the success model is non-trivial and an incorrect model could lead
to inaccurate answers (as a simple example, a model that considers trials to
be perfect would reduce to classical binary search, which as already discussed
is problematic). The benchmark script also supports a "calibration" mode, in
which every candidate rate is tested a large number of times and the fraction
of successes is printed. This does not automatically feed this information
back into the (hard-coded) model, but it provides some data to guide the
developer in writing or updating the model.

.. option:: --calibrate

   Run the calibration mode instead of the usual search

.. option:: --calibrate-repeat <N>

   Set the number of repetitions for each rate.

It is highly recommended that :option:`--low`, :option:`--high` are used to specify
a much smaller range around the critical rate, as this process is extremely
slow.
