"""Offline inference config/checkpoint checks; never connects to a policy or robot."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from config.inference_config import load_inference_config, configure_execution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inference_config')
    parser.add_argument('--checkpoint-config', help='Checkpoint/server metadata JSON to validate.')
    parser.add_argument('--schema-only', action='store_true', help='Validate choices without verifying a checkpoint.')
    args = parser.parse_args(argv)
    try:
        task, resolved = load_inference_config(args.inference_config)
        verified = False
        if not args.schema_only:
            path = args.checkpoint_config or resolved.get('_inference_checkpoint')
            if not path:
                raise ValueError('Supply --checkpoint-config for remote metadata, or --schema-only for config inspection')
            with open(path) as stream:
                metadata = json.load(stream)
            task = configure_execution(resolved, task, metadata, backend=resolved['_inference_document']['backend'])
            verified = True
        print(json.dumps(dict(
            task=task.name, checkpoint_verified=verified, contract=task.resolved(),
            arguments={key: value for key, value in resolved.items() if not key.startswith('_')}), indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
