"""Compatibility import for visualization.visualization_utils.rerun_utils."""
import sys
from visualization.visualization_utils import rerun_utils as _implementation

sys.modules[__name__] = _implementation
