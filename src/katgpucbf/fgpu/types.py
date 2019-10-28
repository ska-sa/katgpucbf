from typing import Union

from katsdpsigproc import opencl, cuda


# TODO: introduce these as real classes in katsdpsigproc
AbstractContext = Union[cuda.Context, opencl.Context]
AbstractCommandQueue = Union[cuda.CommandQueue, opencl.CommandQueue]
AbstractEvent = Union[cuda.Event, opencl.Event]

