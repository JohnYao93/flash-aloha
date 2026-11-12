# Source this file from any working directory.
FLASH_ALOHA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SDK_ROOT="$FLASH_ALOHA_ROOT/src/hardware/arx5-sdk"
export PYTHONPATH="$FLASH_ALOHA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Prefer the CMake-selected runtime libs so Ubuntu 20.04 hosts use the
# compatibility copies when the SDK ships multiple glibc variants.
if [ -d "$SDK_ROOT/build_uv/resolved_libs" ]; then
export LD_LIBRARY_PATH="$SDK_ROOT/build_uv/resolved_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [ -d "$SDK_ROOT/build/resolved_libs" ]; then
export LD_LIBRARY_PATH="$SDK_ROOT/build/resolved_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
export LD_LIBRARY_PATH="$SDK_ROOT/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$SDK_ROOT/lib/x86_64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# The ARX SDK's Conda environment (src/hardware/arx5-sdk/README.md) supplies its native
# libraries. Set ARX_CONDA_PREFIX if it is not at the default Miniconda location.
ARX_CONDA_PREFIX="${ARX_CONDA_PREFIX:-$HOME/miniconda3/envs/arx-py310}"
export LD_LIBRARY_PATH="$ARX_CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AMENT_PREFIX_PATH="${AMENT_PREFIX_PATH:+$AMENT_PREFIX_PATH:}$ARX_CONDA_PREFIX/"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SDK_ROOT/python"
# PyKDL built from source (only if the environment does not already provide it).
# Set OROCOS_KDL_PYTHON to that build directory if it is elsewhere.
OROCOS_KDL_PYTHON="${OROCOS_KDL_PYTHON:-$HOME/orocos_kinematics_dynamics/python_orocos_kdl/build}"
if [ -d "$OROCOS_KDL_PYTHON" ]; then
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$OROCOS_KDL_PYTHON"
fi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$FLASH_ALOHA_ROOT/src/kdl_parser/kdl_parser_py"
