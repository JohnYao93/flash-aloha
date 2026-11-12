#!/usr/bin/env python3
"""Policy inference: local, remote client, or model server.

    python scripts/infer.py direct --inference-config single_arm_flow
    python scripts/infer.py serve --config ckpts/single_arm_flow/all_configs.json --device cuda:0
    python scripts/infer.py remote --inference-config single_arm_remote --remote_host SERVER_HOST

scripts/run.py resolves these arguments from a training-config name.
"""
from _launcher import main

MODES = {
    'direct': ('src/inference/inference_direct.py', 'load the checkpoint and run the robot in this process'),
    'remote': ('src/inference/inference_remote.py', 'robot-side client of a policy server'),
    'serve': ('src/inference/serve_policy.py', 'serve a checkpoint over WebSocket to a remote client'),
}

if __name__ == '__main__':
    main(__doc__.splitlines()[0], MODES)
