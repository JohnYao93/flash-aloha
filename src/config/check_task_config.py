#!/usr/bin/env python3
"""Resolve task YAML and optionally check saved artifacts, without hardware."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.task_config import ConfigLoader, load_task_config
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--recording', type=Path, action='append', default=[])
    parser.add_argument('--checkpoint-config', type=Path)
    parser.add_argument('--legacy-recording-fps', type=float)
    args = parser.parse_args()
    overrides = {}
    for assignment in args.set:
        key, separator, value = assignment.partition('=')
        if not separator or not all(key.split('.')):
            parser.error('--set requires a dotted KEY=VALUE')
        current = overrides
        for part in key.split('.')[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                parser.error(f'Conflicting override: {key}')
            current = child
        leaf = key.split('.')[-1]
        if leaf in current:
            parser.error(f'Duplicate override: {key}')
        current[leaf] = yaml.load(value, Loader=ConfigLoader)
    config = load_task_config(args.config, overrides=overrides)
    for path in args.recording:
        if path.suffix == '.json':
            metadata = json.loads(path.read_text())
            # Raw manifests declare FPS at the root, not in camera sidecars.
            config.require_recording(metadata, legacy_fps=args.legacy_recording_fps)
        else:
            from data_processing import episode_store
            with episode_store.Store(path, 'r') as root:
                config.require_recording(root.attrs, legacy_fps=args.legacy_recording_fps)
    if args.checkpoint_config:
        # The checkpoint's training config supplies the predicted chunk size.
        config = config.require_checkpoint(json.loads(args.checkpoint_config.read_text()))
    print(json.dumps(config.resolved(), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
