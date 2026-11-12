### Task parameters
import os

from config.paths import DATA_DIR as _DATA_DIR

DATA_DIR = str(_DATA_DIR)
ARX_SINGLE_ARM_CAN_BY_SIDE = {
    'left': 'can0',
    'right': 'can1',
}


def resolve_arx_single_arm_side(task_config):
    """Return validated ARX single-arm side, or None for legacy configs."""
    side = task_config.get('single_arm_side')
    if side is None:
        return None
    if side not in ARX_SINGLE_ARM_CAN_BY_SIDE:
        valid = ', '.join(sorted(ARX_SINGLE_ARM_CAN_BY_SIDE))
        raise ValueError(f"single_arm_side must be one of {valid}; got {side!r}")
    return side


def resolve_arx_single_arm_can(task_config):
    """Resolve active CAN for ARX single-arm mode."""
    if task_config.get('_task_config_path'):
        return task_config['can_interface_left']
    side = resolve_arx_single_arm_side(task_config)
    if side is None:
        return task_config.get('can_interface_left', 'can0')
    return ARX_SINGLE_ARM_CAN_BY_SIDE[side]


# YAML is authoritative; preserve the mapping API for existing consumers.
from config.task_config import task_catalog
TASK_CONFIGS = task_catalog()

# Compatibility defaults for helpers without a selected task.
EXECUTION_DT = 0.01
FPS = 100
