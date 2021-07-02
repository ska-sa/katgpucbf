# noqa: D104
from ._katfgpu import Stopped, Empty

try:
    from .version import version

    __version__ = version
except ImportError:
    # Something went wrong because .version should have been generated on install.
    __version__ = "unknown"

__all__ = ["Stopped", "Empty"]
