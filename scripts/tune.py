#!/usr/bin/env python3
"""Gravity and friction compensation tuning: the first step with a newly set up arm.

    uv run scripts/tune.py gravity L5 can0              # link masses
    uv run scripts/tune.py friction L5 can0             # gripper and joint friction terms

Tune gravity first, then friction, once per physical arm. Every argument after the
mode passes to the ARX SDK tuner unchanged (MODEL INTERFACE). Live tuning enables the
motors: support the arm before starting and before exiting.
"""
import os
from pathlib import Path
import sys

import _launcher

# ARX SDK tuners (arguments: MODEL INTERFACE).
MODES = {
    'gravity': ('src/hardware/arx5-sdk/python/examples/tune_gravity_compensation.py',
                'tune link masses until the arm floats'),
    'friction': ('src/hardware/arx5-sdk/python/examples/tune_gripper_friction_compensation.py',
                 'tune gripper and joint friction feedforward'),
}
def command(argv):
    """The interpreter command for MODE ARGS: the ARX SDK tuner with ARGS unchanged."""
    return _launcher.command(MODES, argv[0], list(argv[1:]))


def usage(prog):
    lines = [f'usage: {prog} MODE [ARGS ...]', '', __doc__.splitlines()[0], '', 'modes:']
    for mode in MODES:
        lines += [f'  {mode:<8}  {MODES[mode][1]}', f'            {MODES[mode][0]}  MODEL INTERFACE']
    lines += ['', f'Run "{prog} MODE --help" for the options of that tuner.']
    return '\n'.join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    prog = Path(sys.argv[0]).name
    if not argv or argv[0] in ('-h', '--help'):
        print(usage(prog))
        sys.exit(0 if argv else 2)
    if argv[0] not in MODES:
        print(usage(prog), file=sys.stderr)
        sys.exit(f'{prog}: unknown mode {argv[0]!r}; choose from {", ".join(MODES)}')
    # Preserve interpreter flags, environment, cwd, exit status and signals.
    os.execv(sys.executable, command(argv))


if __name__ == '__main__':
    main()
