#!/usr/bin/env python3
"""Run the implementation in src/camera_utils/camera_int.py."""
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    implementation = root / "src/camera_utils/camera_int.py"
    interpreter_args = sys.orig_argv[1:len(sys.orig_argv) - len(sys.argv)]
    os.execv(sys.executable, [sys.executable, *interpreter_args, str(implementation), *sys.argv[1:]])
