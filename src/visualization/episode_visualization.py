"""Compatibility import for visualization.visualization_utils.episode_visualization."""
import sys
from visualization.visualization_utils import episode_visualization as _implementation

sys.modules[__name__] = _implementation
