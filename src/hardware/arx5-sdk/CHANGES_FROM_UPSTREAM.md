# Changes from upstream arx5-sdk

This directory is a modified copy of
[real-stanford/arx5-sdk](https://github.com/real-stanford/arx5-sdk), based on commit
[`79240efc`](https://github.com/real-stanford/arx5-sdk/tree/79240efc2e0d75fdc1e50a8b883c5e928fbfc681).
The changes below are grouped as units that could each become one upstream pull
request. Paths are relative to this directory. The vendor libraries in `lib/` and the
low-level CAN headers are unchanged.

## Candidate pull requests

| # | Change | Files | Notes for upstreaming |
|---|---|---|---|
| 1 | **Cubic interpolation fix.** Corrects the Hermite duration factors for position and velocity, plus loop bounds and failure handling. | `src/utils.cpp` | Self-contained bug fix. Only affects cubic trajectories with non-unit segment durations; the default stays linear. |
| 2 | **Startup, shutdown and emergency locking.** Background communication starts after the derived controller is initialised; shutdown is an idempotent helper; the Cartesian destructor stops the controller first; state/command locks are released before entering emergency. | `src/app/controller_base.cpp`, `src/app/cartesian_controller.cpp` | Self-contained robustness fixes. |
| 3 | **Per-model velocity scaling.** Decoded joint velocities are scaled by 10/45 (joints 1–3) and 30/45 (joints 4–6 and gripper); outgoing DaMiao velocity commands use the inverse. | `include/app/config.h`, `include/app/common.h`, `src/app/controller_base.cpp` | Applied to `L5` and `Trossen_L5` only. Upstream would need to confirm the motor firmware ranges this assumes. |
| 4 | **Friction feedforward.** A `<arx5_friction_compensation>` block in the URDF (constant, Coulomb with `tanh`, viscous terms per joint and for the gripper), loaded by a new `get_config("joint_controller", robot_config)` overload, with runtime switches for gravity, joint friction, gripper friction and a master switch. | `src/app/friction_config.cpp` (new), `include/app/config.h`, `src/app/controller_base.cpp`, `python/arx5_pybind.cpp`, `python/arx5_interface.pyi`, `CMakeLists.txt` (TinyXML2) | The old `get_config(..., joint_dof)` overload is kept and leaves friction off. Depends on TinyXML2. |
| 5 | **Gravity and friction tuners.** Interactive tools that edit link masses or friction terms in the selected URDF. | `python/examples/tune_gravity_compensation.py`, `python/examples/tune_gripper_friction_compensation.py`, `python/examples/test_gripper_friction_compensation.py`, `examples/test_gripper_friction_compensation.cpp` | Needs change 4. |
| 6 | **Cartesian torque control.** Optional six-joint Cartesian PD (host Cartesian damping, or host stiffness with motor-local joint damping), Cartesian trajectory APIs, and safeguards (wrench limits, tracking limits and timeout, joint-limit margin). Disabled by default with zero gains. | `include/app/cartesian_pd.h`, `src/app/cartesian_pd.cpp`, `src/app/cartesian_ik.{h,cpp}` (new); `src/app/cartesian_controller.cpp`, `include/app/cartesian_controller.h`; tools `python/examples/tune_cartesian_pd.py`, `record_cartesian_pd_motion.py`, `record_replay_hybrid_cartesian.py`, `simulate_cartesian_pd.py`; `tests/` | Largest change. Also changes the legacy IK path: it seeds from the measured pose first and rejects invalid or out-of-limit solutions. |
| 7 | **Host joint damping and control-cycle telemetry.** A host-side damping vector (zero by default) and a per-cycle snapshot of feedback, references, torques, gains and send timing, exposed to Python. | `src/app/joint_controller.cpp`, `include/app/common.h`, `src/app/controller_base.cpp`, `python/arx5_pybind.cpp`; tools `python/examples/record_torque_latency.py`, `compare_torque_latency_ab.py` | Independent of change 6 apart from the shared telemetry fields. |
| 8 | **Build.** A library resolver that picks Ubuntu 20.04-compatible vendor libraries on older glibc, explicit SOEM resolution, the namespaced KDL parser target, and C++ examples/tests behind `ARX5_BUILD_CPP_TESTS` (default off). | `CMakeLists.txt`, `python/CMakeLists.txt`, `cmake/SelectSdkLibDir.cmake` (new), `env.sh` | `env.sh` contains a machine-specific Conda path; generalise before upstreaming. |
| 9 | **Gripper sanity tolerance.** The accepted closed-side reading widens from −5 mm to −10 mm. | `src/app/controller_base.cpp` | One-line change; the open-side tolerance and command clipping are unchanged. |
| 10 | **Calibration script options.** Joint calibration can return immediately or print with `--watch` / `--watch-seconds`, and disables gravity and friction while inspecting. | `python/examples/calibrate.py` | Running the script directly now calls `calibrate_gripper()` (upstream calls `check_motor_movements()`); revert that default before upstreaming. |

## Local values, not for upstreaming

These are settings for the arms this repository was developed on.

| Item | Files | What differs |
|---|---|---|
| `Trossen_L5` model | `include/app/config.h`, `models/Trossen_L5.urdf`, `models/Trossen_L5_cartesian_pd.json` | A separate profile: gripper width 0.056 m and open readout 3.03 rad, higher Cartesian speed limits, its own link masses and friction terms. |
| `L5` gripper calibration | `include/app/config.h` | Open width 0.100 m (upstream 0.088 m) and open readout 5.1276 rad (upstream 5.03 rad). |
| `L5` link masses and friction terms | `models/L5.urdf` | Masses of links 2–6 retuned; a friction block added. Written by the tuners in change 5. |
| MuJoCo model | `models/mujoco/ARX_L5/` | Used only by `python/examples/simulate_cartesian_pd.py`. |

## Removed from this copy

Upstream files that this copy does not carry: the SpaceMouse teleoperation example and
reader, the ZMQ server/client and example, the `python/shared_memory` helpers, and
`update_logs.md`.

## Upstream changes not merged here

Upstream has since fixed blocked-gripper direction detection for arms whose
`gripper_open_readout` is negative (it multiplies the torque by the readout sign). This
copy still uses the torque alone, which gives the same result for the positive readouts
configured here. Rebase onto that fix before opening pull requests.
