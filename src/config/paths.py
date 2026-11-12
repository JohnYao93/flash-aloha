"""Storage locations: datasets, checkpoints and logs.

Defaults are ``data/``, ``ckpts/`` and ``logs/`` in the repository root. Set
``FLASH_ALOHA_DATA_DIR``, ``FLASH_ALOHA_CKPT_DIR`` or ``FLASH_ALOHA_LOG_DIR`` to keep
them elsewhere (for example on a larger disk). Task, training and inference YAMLs keep
their repository-relative paths (``../../../data/TASK``, ``../../../ckpts``): every such
path is resolved through ``storage_path``, which maps the default locations to the
configured ones, so one setting moves everything consistently. Weights & Biases uses its
own ``WANDB_DIR``.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / 'data'
DEFAULT_CKPT_DIR = REPO_ROOT / 'ckpts'
DEFAULT_LOG_DIR = REPO_ROOT / 'logs'


def _configured(variable, default):
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _configured('FLASH_ALOHA_DATA_DIR', DEFAULT_DATA_DIR)
CKPT_DIR = _configured('FLASH_ALOHA_CKPT_DIR', DEFAULT_CKPT_DIR)
LOG_DIR = _configured('FLASH_ALOHA_LOG_DIR', DEFAULT_LOG_DIR)
ROLLOUT_DIR = DATA_DIR / 'rollouts'
DAGGER_DIR = DATA_DIR / 'dagger_data'
INFERENCE_RAW_DIR = DATA_DIR / 'inference_raw'
INFERENCE_LOG_DIR = LOG_DIR / 'inference'  # per-run direct inference console and step logs
REPLAY_LOG_DIR = LOG_DIR / 'replay'  # per-take physical replay commanded/measured step logs


def storage_path(path):
    """An absolute path, with the repository's default data/ckpts/logs locations mapped to the configured ones."""
    path = Path(os.path.normpath(Path(path).expanduser()))  # no symlink resolution yet
    if not path.is_absolute():
        raise ValueError(f'storage_path needs an absolute path, got {path}')
    for default, configured in ((DEFAULT_DATA_DIR, DATA_DIR), (DEFAULT_CKPT_DIR, CKPT_DIR), (DEFAULT_LOG_DIR, LOG_DIR)):
        if configured != default and (path == default or default in path.parents):
            path = configured / path.relative_to(default)
            break
    return path.resolve()
