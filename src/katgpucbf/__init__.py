# noqa: D104
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as version_func
from typing import Final

try:
    __version__ = version_func(__name__)
except PackageNotFoundError:
    # Package wasn't installed yet?
    __version__ = "unknown"

__all__ = ["__version__"]

CPLX: Final[int] = 2
N_POLS: Final[int] = 2
