"""Mode dispatch shared by scripts/infer.py, record.py and visualize.py.

Each mode runs one unchanged entry point under src/ with the remaining
arguments passed through as-is; ``MODE --help`` shows that entry point's options.
"""
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def command(modes, mode, args):
    """The interpreter command for MODE: the src/ entry point plus ARGS, unchanged."""
    path, _ = modes[mode]
    interpreter_args = sys.orig_argv[1:len(sys.orig_argv) - len(sys.argv)]
    return [sys.executable, *interpreter_args, str(REPO_ROOT / path), *args]


def usage(prog, summary, modes):
    width = max(map(len, modes))
    lines = [f'usage: {prog} MODE [ARGS ...]', '', summary, '', 'modes:']
    lines += [f'  {mode:<{width}}  {description}  ({path})' for mode, (path, description) in modes.items()]
    lines += ['', f'Run "{prog} MODE --help" for the options of that mode.']
    return '\n'.join(lines)


def main(summary, modes, argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    prog = Path(sys.argv[0]).name
    if not argv or argv[0] in ('-h', '--help'):
        print(usage(prog, summary, modes))
        sys.exit(0 if argv else 2)
    if argv[0] not in modes:
        print(usage(prog, summary, modes), file=sys.stderr)
        sys.exit(f'{prog}: unknown mode {argv[0]!r}; choose from {", ".join(modes)}')
    # Preserve interpreter flags, environment, cwd, exit status and signals.
    args = command(modes, argv[0], argv[1:])
    os.execv(sys.executable, args)
