from typing import Union

from katsdpsigproc import cuda


# TODO: introduce these as real classes in katsdpsigproc
AbstractContext = Union[cuda.Context]
AbstractCommandQueue = Union[cuda.CommandQueue]
AbstractEvent = Union[cuda.Event]
