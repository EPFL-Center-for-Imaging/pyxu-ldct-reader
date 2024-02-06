import importlib.metadata

try:
    __version__ = importlib.metadata.version("pyxu_eigh")
except ImportError:
    __version__ = "unknown"


__all__ = ()
