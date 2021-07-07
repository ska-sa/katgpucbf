# noqa: D104
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as version_func

from ._katfgpu import Empty, Stopped

try:
    __version__ = version_func(__name__)
except PackageNotFoundError:
    # Package wasn't installed yet?
    __version__ = "unknown"

__all__ = ["Stopped", "Empty"]
