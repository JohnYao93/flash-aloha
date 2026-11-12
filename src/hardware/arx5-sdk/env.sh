#!/usr/bin/env bash

SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "$SDK_ROOT/build_uv/resolved_libs" ]; then
  export LD_LIBRARY_PATH="$SDK_ROOT/build_uv/resolved_libs:$LD_LIBRARY_PATH"
elif [ -d "$SDK_ROOT/build/resolved_libs" ]; then
  export LD_LIBRARY_PATH="$SDK_ROOT/build/resolved_libs:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$SDK_ROOT/lib/aarch64:$LD_LIBRARY_PATH"
  export LD_LIBRARY_PATH="$SDK_ROOT/lib/x86_64:$LD_LIBRARY_PATH"
fi

export LD_LIBRARY_PATH="$HOME/miniconda3/envs/arx-py310/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$SDK_ROOT/python:$PYTHONPATH"
