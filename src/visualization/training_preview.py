"""Compatibility import for visualization.visualization_utils.training_preview."""
import sys
from visualization.visualization_utils import training_preview as _implementation

sys.modules[__name__] = _implementation
