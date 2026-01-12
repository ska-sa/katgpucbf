Updating autotuning database
============================

Some kernels having tuning parameters which can be optimised automatically.
:external+katsdpsigproc:doc:`Support for this <user/autotune>` is provided by the
katsdpsigproc library.

To avoid autotuning occurring at startup, a database of pre-tuned parameters
is stored in the repository and inserted into the Docker image. If necessary,
it can be updated using the script :file:`docker/autotune.py`, run on a
machine with a suitable GPU. It does not need to exactly match the GPU used
for deployment, but the more similar it is, the better the tuning will be.

To ensure that the tuning is done with the same environment (particularly CUDA
compiler version) that will be deployed, the autotuning should be run inside
the Docker image. This can be done something like this (the Docker image path
is just an example; adjust as necessary), and takes roughly 5 hours::

    docker pull harbor.sdp.kat.ac.za/dpp/katgpucbf
    docker run -it --rm --gpus=all -v $PWD/docker:/output harbor.sdp.kat.ac.za/dpp/katgpucbf /output/autotune.py /output/tuning.db

Note that this may cause the database (:file:`docker/tuning.db`) to be
owned by root and require fixing. Check that the database is non-empty, then
commit it.

The tuning script covers a reasonably large range of parameters, but cannot
cover everything. If you see run-time log messages indicating that autotuning
is occurring, you may need to revise the script.
