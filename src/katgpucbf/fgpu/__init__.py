# noqa: D104
from ._katfgpu import Stopped, Empty

from importlib.metadata import PackageNotFoundError, version as version_func

try:
    __version__ = version_func(__name__)
except PackageNotFoundError:
    # Package wasn't installed yet?
    __version__ = "unknown"

__all__ = ["Stopped", "Empty"]
