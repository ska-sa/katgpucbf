XB-Engine design
================

Similar to the F-engine, the XB-engine supports receiving one set of input and
producing multiple output streams. The incoming data is received by the engine
itself, and then distributed to individual
:class:`~katgpucbf.xbgpu.engine.Pipeline` instances for processing and
transmission of results.

.. toctree::

   xbgpu.design.x
   xbgpu.design.b
