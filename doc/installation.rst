Installation
============

Installation with Docker
------------------------
The recommended way to use katgpucbf is via Docker. There is currently no
published Docker image, so it is necessary to build your own. To do so, change
to the root directory of the repository and run

.. code:: sh

   docker build -t NAME .

where :samp:`{NAME}` is the name to assign to the image.

You will need to have the NVIDIA container runtime installed to provide Docker
with access to the GPU.

Installation with pip
---------------------
It is also possible to install katgpucbf with pip. In this case, you will need
to have CUDA already installed. Change to the root directory of the repository
and run

.. code:: sh

   pip install ".[gpu]"

Note that if you are planning to do development on katgpucbf, you should refer
to the :doc:`Developers' guide <dev-guide>`.
