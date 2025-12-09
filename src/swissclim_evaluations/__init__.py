import matplotlib

# Force non-interactive backend to avoid QSocketNotifier/threading issues
matplotlib.use("Agg")

from . import data, helpers  # re-exported for convenience

__all__ = ["data", "helpers"]

# Optional package metadata
__version__ = "0.1.0"
