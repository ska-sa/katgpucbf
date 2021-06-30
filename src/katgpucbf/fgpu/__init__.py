# noqa: D104
from ._katfgpu import Stopped, Empty
from importlib.metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("katfgpu")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["Stopped", "Empty"]
