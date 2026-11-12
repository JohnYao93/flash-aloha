> This directory is a copy of [real-stanford/arx5-sdk](https://github.com/real-stanford/arx5-sdk) with local
> modifications (see [CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md)). The original README follows.

# C++ && Python Controller for ARX5 Robot Arm

## Safety-Related Configs

 - The safety checks in this controller are implemented through joint position, velocity, and torque limits. However, there is a trade-off between safety and reactiveness when setting these limits. To achieve a reasonable precision for our experiements, **the default values are insufficient to guarantee safety when the robot is close to singularity or the input control signal includes a lot of noise**.
- We highly recommend that users should **apply safety checks before sending control signals** (joint/eef) to the controller
- Users could also modify the limit values of the joint velocity at their early stage of deployment. Please change the config values before instantiating the controller in python, similar to [this example](https://github.com/real-stanford/arx5-sdk/blob/709f7ab7429f97c83e18687e650f3ee77d14719a/python/examples/test_joint_control.py#L31). You need to set the values with a new numpy array, e.g. `robot_config.joint_vel_max=np.array([2,2,2,2,2,2])`, rather than indexing some of the existing values `robot_config.joint_vel_max[0]=2.0`, which will raise an error.

## Major Update (2026.03.20)
- Enable direct pip install for both `x86_64` and `aarch64` platforms, supporting python 3.8~3.14. You may use `pip install arx5-interface` to install the package without any conda / system dependencies. If you need to updated any C++ files, you may also run `wheels/build_wheel_single_ver.sh` to build the wheel for your specific python version and directly install this wheel in pip or uv.
- Set the gripper home position to be the fully opened position. To keep using the closed position as home, you may set the `target_state.gripper_pos` to 0 in `src/app/controller_base.cpp:187`.
- Use relative path for the `urdf_path` in `include/app/config.h` based on the dynamic library loading path. Thanks [Jimmy Wu](https://jimmyyhwu.github.io/) for the PR.

## Update (2026.01.11)
- In some recent robot models, the gripper motor is mounted in a different direction, which will lead to error: "Gripper position error: got xxx but should be in xxx". Please set the `robot_config.gripper_open_readout` to a negative number and it should work normally. According to a github issue, a typical readout might be `gripper_open_readout=-3.4` and `robot_config.gripper_width=0.082`. If the readout gripper width is inaccurate, you may also calibrate yourself in `python/examples/calibrate.py`

## Features
- Run without ROS
- No `sudo` requirement for building the library (all dependencies are managed under conda environment, thanks to [Cheng Chi](https://cheng-chi.github.io/))
- Simple python interface with complete type hints (see `python/arx5_interface.pyi`, please include it into your vscode config `"python.analysis.extraPaths"`)
- Joint controller runs at 500Hz in the background (motor communication delay ~0.4ms)
- Cartesian space controller with keyboard tele-operation and teach-replay (thanks to [Cheng Chi](https://cheng-chi.github.io/))
- Control multiple arms in the same process through C++ multi-threading (much better than Python multi-processing)

### Joint-PD friction compensation

The model-aware `Arx5JointController` configuration enables both arm-joint
(body) and gripper friction feedforward by default when calibrated terms are
present in the model URDF, as they are for `L5` and `Trossen_L5`. This applies
to the direct `Arx5JointController(model, interface)` constructor and to
`get_config("joint_controller", robot_config)`. Cartesian/hybrid controller
configurations remain disabled by default. Runtime setters can still disable
the body, gripper, or master friction switch independently.

### Cartesian PD control

Six-joint X5 and L5 variants support optional operational-space PD in
`Arx5CartesianController`. Cartesian PD is disabled by default, so existing joint-PD
behavior is unchanged. The six gain axes are ordered `[x, y, z, rx, ry, rz]`.
Translational `kp` is in N/m, translational `kd` is in N/(m/s), rotational `kp` is
in Nm/rad, and rotational `kd` is in Nm/(rad/s). Both gain vectors default to
zero so enabling Cartesian mode does not introduce stiffness before gains are
chosen explicitly. The default per-axis wrench limits remain
`[30,30,30,5,5,5]` in N and Nm.

Cartesian mode maps the clamped task-space wrench through the measured Jacobian
transpose. Gravity compensation remains configurable, and final motor commands
still use the robot's per-joint torque limits. In the default damping mode,
arm motor-level joint gains are zero; gripper gains are unchanged.

Cartesian damping defaults to `CartesianDampingMode.HostCartesian`, preserving
that host-side P+D behavior. `MotorLocalJoint` retains Cartesian stiffness on
the host while moving damping into the drives:

```python
controller.set_cartesian_damping_mode(arx5.CartesianDampingMode.MotorLocalJoint)
controller.set_cartesian_motor_kd(np.array([5, 5, 5, 1, 1, 1], dtype=float))
controller.enable_cartesian_pd()
```

Hybrid commands force arm motor Kp to zero, use the configured motor Kd, and
derive the motor velocity target from the desired Cartesian twist with a damped
pseudoinverse. Mode and motor-Kd changes are rejected while Cartesian PD is
enabled. Cycle telemetry includes the final packed gains and protocol velocity,
host stiffness/damping and gravity torques, velocity limiting, and final `t_ff`.

While Cartesian PD is enabled, `set_eef_cmd()` and `set_eef_traj()` generate
pose and analytic twist references directly in Cartesian space at the 500 Hz
controller rate; live IK and finite-differenced joint targets are not in this
path. Pose-only commands stop at their final waypoint. Moving sources can use
`set_eef_cmd_with_twist(...)` or `set_eef_traj_with_twists(...)`, with base-frame
twist order `[vx, vy, vz, wx, wy, wz]`, and inspect the active reference with
`get_eef_cmd_twist()`. When Cartesian PD is disabled, the original IK-to-joint-PD
behavior remains available for compatibility.

Activation anchors at the measured pose, Cartesian task torque ramps in over at
least 20 ms, and reference speed is bounded by `RobotConfig.ee_vel_max`. A 100 mm
or 45 degree tracking error freezes and cancels reference progression; failure
to recover within 250 ms disables Cartesian PD. These guards do not check
self-collision or external obstacles.

Cartesian damping can optionally use an internal causal first-order low-pass
filter on the measured joint velocity:

```text
alpha_k = 1 - exp(-2 pi f_c delta_t_k)
qdot_filtered_k = qdot_filtered_(k-1)
                  + alpha_k (qdot_measured_k - qdot_filtered_(k-1))
```

All Cartesian configurations default to `f_c = 0 Hz`, which exactly bypasses
the filter. A positive cutoff enables filtering and must remain below the
controller Nyquist frequency (250 Hz at 500 Hz). The filter uses measured
sample timestamps, initializes from the first sample, and resets after invalid
timing or a gap longer than ten controller periods. Only the measured velocity
used by Cartesian Kd is filtered: desired velocity, public joint state,
joint-space control, logging, and friction compensation remain unchanged.

At the 500 Hz controller rate, an optional 50 Hz cutoff has a step time
constant of about 3.2 ms and approximately 2.3 ms of low-frequency
discrete-time delay. Set a positive cutoff before construction with
`controller_config.cartesian_joint_velocity_filter_cutoff_hz`, or change it at
runtime with `set_cartesian_joint_velocity_filter_cutoff(...)`. Use
`get_filtered_joint_velocity()` to inspect the internal Cartesian feedback
signal.

```python
gain = arx5.CartesianPDGain(
    np.array([100, 100, 100, 10, 10, 10], dtype=np.float64),
    np.array([20, 20, 20, 2, 2, 2], dtype=np.float64),
)
controller.set_cartesian_pd_gain(gain)
controller.enable_cartesian_pd()  # freezes the desired pose at the measured pose
```

Use conservative targets, stay away from singularities, and keep an emergency
stop available. Wrench limiting does not guarantee safe contact force, especially
near singular configurations. Enabling Cartesian PD on X7 or another non-six-joint
model raises an unsupported-model error.

Tune gains interactively from zero at the measured end-effector pose:

```sh
PYTHONPATH=python python python/examples/tune_cartesian_pd.py L5 can0
```

The tuner changes one Kp/Kd axis at a time, can re-anchor the hold pose, shows
raw, filtered, and desired Cartesian twists, and cycles the cutoff through
`0, 5, 8, 10, 15, 20, 30, 50` Hz with `v`. It saves gains and the cutoff to
`models/L5_cartesian_pd.json` with `s`. Older JSON files without the cutoff use
the model default. The saved file is loaded automatically by the restrained
hardware test:

```sh
PYTHONPATH=python python python/examples/impedance_hold_test.py L5 can0 \
    --target-pose 0.30 0.00 0.25 0.00 0.00 0.00
```

The hold test refuses to move if its movement Kp is still all zero. Gains can
also be supplied directly with `--kp`, `--kd`, `--hold-kp`, and `--hold-kd`;
both tools accept `--filter-cutoff-hz` as an explicit override.

To diagnose motion smoothness, record the same preflighted sinusoid at a 500 Hz
telemetry sampling target while continuing to update its source command at the
tuner's 50 Hz rate:

```sh
PYTHONPATH=python python python/examples/record_cartesian_pd_motion.py \
    L5 can0 --axis rx
```

The hybrid-versus-motor-local stiffness comparison supports all existing
Cartesian test axes. For example:

```bash
PYTHONPATH=python python python/examples/record_cartesian_pd_motion.py \
  Trossen_L5 can2 --axis rx --paired-comparison
```

Replace `rx` with `x`, `y`, `z`, `ry`, or `rz` for the other oscillation
directions. Run one command at a time so the manual positioning, stationary
check, swept-envelope preflight, and operator confirmation happen for every
axis.

It runs hybrid Cartesian stiffness first, returns along the commanded trajectory
to the original test anchor, verifies the 0.05 rad, 5 mm, and 3 degree anchor
tolerances, and switches directly to motor-local joint stiffness without going
home or asking for repositioning. The robot returns home only after the
motor-local condition. Each condition and the combined neutral comparison
receive CSV, JSON, and dependency-free HTML artifacts.

To teach a five-second trajectory by hand and replay its FK path with hybrid
Cartesian impedance:

```bash
PYTHONPATH=python python python/examples/record_replay_hybrid_cartesian.py \
  Trossen_L5 can2 \
  --cartesian-kp 200 200 200 16 16 16
```

The stiffness order is `[x, y, z, rx, ry, rz]`; translation values are N/m and
rotation values are Nm/rad. During teaching, arm gains and friction compensation
are off while gravity compensation remains active. The script returns to the
recorded starting joints using the controller's standard stiff joint gains,
settles for one second, and holds that position while calculating FK poses and
Jacobian twists. The default replay-start tolerances are 0.10 rad per joint,
30 mm in TCP translation, and 10 degrees in orientation; override them with
`--start-joint-tolerance-rad`, `--start-translation-tolerance-mm`, and
`--start-rotation-tolerance-deg`. The script then requests confirmation before
hybrid replay. It writes the raw joint recording, FK reference, replay telemetry,
and JSON metadata alongside the selected output prefix.

To replay the taught joints directly instead of converting them to a Cartesian
reference, select joint mode:

```bash
PYTHONPATH=python python python/examples/record_replay_hybrid_cartesian.py \
  Trossen_L5 can2 --replay-mode joint
```

Joint mode uses the same teach and return-to-start procedure, then submits the
recorded position knots with the controller's standard stiff joint gains. The
single `--trajectory-rate` option (100 Hz by default; `--record-rate` is an
alias) controls both recording and replay knots. Python does not upsample the
positions: the native joint or Cartesian interpolator evaluates them in the
500 Hz controller loop. Joint mode skips FK and Cartesian PD setup and writes
`_joint_reference.csv`, atomic 500 Hz desired/measured `_joint_replay.csv`
telemetry, and JSON metadata. The default mode remains `hybrid-cartesian`.

For a same-recording, same-timing comparison that runs hybrid first and direct
joint replay second:

```bash
PYTHONPATH=python python python/examples/record_replay_hybrid_cartesian.py \
  Trossen_L5 can2 --replay-mode comparison \
  --cartesian-kp 200 200 200 16 16 16
```

The arm returns to the recorded start and validates the common anchor between
conditions. Comparison mode uses the path-median diagonal of
`Kq = J.T @ diag(Kx) @ J` as the direct joint Kp by default and uses the same
motor Kd as hybrid mode. Override the approximation with
`--joint-replay-kp J1 J2 J3 J4 J5 J6`. Since the full projected stiffness is
pose-dependent and contains off-diagonal coupling, fixed diagonal motor gains
are only a rough equivalent. Separate condition telemetry and time-aligned
`_comparison.csv`, `.json`, and `.html` artifacts are written.

The recorder loads the tuner's model-specific gain file by default and writes
`.csv`, `.json`, and dependency-free `.html` files. The HTML report compares
source, 500 Hz desired, and measured Cartesian pose/twist; measured and command
joint position/velocity/torque; and recorder timing. Missed 2 ms sampling
deadlines and command-call duration are retained so host-side recording jitter
is distinguishable from controller or actuator motion. Before anchoring and
preflight, the recorder enters a manual positioning phase with arm PD and
friction compensation off and gravity compensation on. Move the arm to the
desired start pose, release it, and press Enter; recording starts only after a
250 ms stationary check. Use `--skip-positioning` only when the arm is already
at the intended start pose.

The supplied L5 MuJoCo model can exercise the same C++ Cartesian PD kernel
without connecting hardware:

```sh
python -m pip install mujoco
PYTHONPATH=python python python/examples/simulate_cartesian_pd.py
# Add --view for real-time visualization or --csv cartesian_pd.csv for a log.
```

## Build & Install

### pip install (Recommended)

You may use `pip install arx5-interface` to install the package without any conda / system dependencies. If you have already build the package inside the conda environment, you need to delete the existing `.so` file under `python` folder to ensure the pip wheel is being used.

If you encounter error "ERROR: No matching distribution found for arx5-interface", please upgrade pip by `pip install --upgrade pip` and try again.

If you need to updated any C++ files, you may also run `wheels/build_wheel_single_ver.sh` to build the wheel for your specific python version and directly install this wheel. 

If you only need to update the config values, you don't need to change any C++ code. Please refer to `python/examples/test_joint_control.py` to use `robot_config` and `controller_config` before instantiating the controller.

### Build from source (Legacy)

We set up a conda environment for all the cmake dependencies, so no system package is required. If you want to run `cmake` and `make` after modifying the C++ source files, please make sure you are under the created conda environment (`arx-py310` etc.).  

We recommend [mamba](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) for creating conda environments, which takes only about 1min. You can also use `conda`, but it takes significantly longer (~10min).

``` sh
mamba env create -f conda_environments/py310_environment.yaml
# if you do not have mamba, you can also use conda, which takes significantly longer
# Currently available python versions: 3.8, 3.9, 3.10, 3.11, 3.12
conda activate arx-py310
mkdir build && cd build
cmake ..
make -j
# At this point, you should be able to run test scripts below.
```
``` sh
# To install the C++ package your system, run:
make install
```

## EtherCAT-CAN setup

Use a USB cable to power the EtherCAT-CAN adapter and an ethernet cable to connect it to your computer. After running `ip a` in your terminal, you should find the interface name, usually `eth.` (existing ethernet port) or `en..........` (additional USB-Ethernet adapters).

Then you should enable the ethernet access of your Python interpreter (usually in your `bin` folder). Note that `which python` usually gives you a symbolic link (say `~/miniforge3/envs/arx-py310/bin/python`) and doesn't work in this case. You need to find out the actual file (usually `python3.x`). 
```sh
mamba activate arx-py310
ls -l $(which python)
sudo setcap "cap_net_admin,cap_net_raw=eip" your/path/to/conda/envs/arx-py310/bin/python3.10
```

To run C++, you need to enable the executable every time after compiling. You also need to update the C++ scripts with the correct interface.
```sh
sudo setcap "cap_net_admin,cap_net_raw=eip" build/test_cartesian_controller
sudo setcap "cap_net_admin,cap_net_raw=eip" build/test_joint_controller
```

## USB-CAN setup

You can skip this step if you have already set up the EtherCAT-CAN adapter.

``` sh
sudo apt install can-utils
sudo apt install net-tools
```

There are 2 popular firmware types of usb-can adapter, `SLCAN` and `candleLight`. After plugging the adapter, you can find out the correct firmware type by the following rules:
- Run `ls /dev/ttyACM*`. If there is a **new** `/dev/ttyACM*` device (where * is a number), this adapter is using `SLCAN` firmware
- Run `ip a`. If there is a **new** `can*` interface (where * is a number), this adapter is using `candleLight` firmware.

### For adapters using SLCAN framework
Get serial number by:
``` sh
udevadm info -a -n /dev/ttyACM* | grep serial
# Replace the * by the actual number if there are multiple ttyACM devices connected to your computer.
```
You will get something like:
```
ATTRS{serial}=="209738924D4D"
ATTRS{serial}=="0000:00:14.0"
```
Then edit CAN rules file:
``` sh
sudo vim /etc/udev/rules.d/arx_can.rules
```
Copy and paste the following, and replace the serial number with yours. If you are registering multiple adapters, you can use other `SYMLINK` names (e.g. `arxcan1`) and make sure the following commands are updated accordingly.
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="209738924D4D", SYMLINK+="arxcan0"
```

Finally, activate CAN connection by: (**the second line should be run every time after connection**)
``` sh
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up
```

Alternatively, if you want not to run the second line everytime, you can also setup a system service:

```sh
sudo vi /etc/systemd/system/arxcan-setup.service
```

Copy the following content to the service file

```service
[Unit]
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'slcand -o -f -s8 /dev/arxcan0 can0 && ip link set can0 up'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

And activate the service

```
sudo systemctl daemon-reload
sudo systemctl enable arxcan-setup.service
sudo systemctl start arxcan-setup.service
sudo systemctl status arxcan-setup.service
```

### For adapters using candleLight framework
After plugging the adapter and running `ip a`, you should immediately find a can interface (usually `can0`). If you only have one arm, simply run 
```sh
sudo ip link set up can0 type can bitrate 1000000
```
and you are good to go. You should run it **every time** after connecting a usb-can adapter.
If you have multiple arms and you want to fix the CAN interface name mapping for each arm, you need to register the adapter to the CAN rules:
```sh
sudo dmesg # Find the idVendor, idProduct and serial number of your adapter
sudo vim /etc/udev/rules.d/arx_can.rules
```
and then append the following line to your `arx_can.rules`. Make sure you've replaced the serial number of your adapter and your desired CAN name.
```
SUBSYSTEM=="net", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", ATTRS{serial}=="0040001E3730511620323746", NAME="can0"
```
Run the following line to update the changes
```sh
sudo udevadm control --reload-rules && sudo systemctl restart systemd-udevd && sudo udevadm trigger
```
Finally, reconnect your adapter and run (**This line should be run every time after plugging the usb-can adapter**)
```sh
sudo ip link set up can0 type can bitrate 1000000
```


## Test scripts

Arguments for `test_joint_control.py`, `keyboard_teleop.py` and `teach_replay.py`: 
- (required) model: `X5` (silver and black) or `L5` (all black metal with blue or red LED light). **Choosing the wrong model may lead to dangerous movements!**
- (required) interface: `can0`, `enx6c1ff70ac436` etc. (run `ip a` to check your interface name)
- (optional) urdf_path `-u`: by default `../models/arx5.urdf`

```bash
cd python
python examples/test_joint_control.py X5 can0 # replace X5 with L5 for the other model 
python examples/test_bimanual.py # For two X5 arms using can0 and can1, each arm will act the same as test_joint_control.py
python examples/keyboard_teleop.py X5 can0
python examples/teach_replay.py X5 can0
```
To use python sdk from other directories, please make sure `./python` is in `$PYTHONPATH` and `./lib/x86_64` or `./lib/aarch64` (depend on your computer system) is in `$LD_LIBRARY_PATH`. 

``` sh
export PYTHONPATH=$PYTHONPATH:/path/to/your/arx5-sdk/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/arx5-sdk/lib/your_arch
```

After compiling the `arx5_interface` pybind dynamic library (usually `python/arx5_interface.cpython-version-arch-linux-gnu.so`), you can run it under other python environments (need to be the same python version as the one you built).
 
## Projects Using this Repository
- **[UMI-on-Legs](https://umi-on-legs.github.io/)**
- [UMI](https://umi-gripper.github.io/): [deployment code](https://github.com/real-stanford/umi-arx)
- [UVA](https://unified-video-action-model.github.io/): [deployment code](https://github.com/real-stanford/umi-arx/tree/uva)
- [Vision-in-Action](https://vision-in-action.github.io/)
- [Latent Policy Barrier](https://latentpolicybarrier.github.io/)
- [DynaGuide](https://dynaguide.github.io/)
- [Minimalist Compliance Control](https://minimalist-compliance-control.github.io/)
- [Gated Memory Policy](http://gated-memory-policy.github.io/)

More cool projects on the way!

## Citation
If you find this repo helpful, please cite our corresponding paper [UMI-on-Legs](https://umi-on-legs.github.io/)
```
@inproceedings{ha2024umilegs,
  title={{UMI} on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers},
  author={Huy Ha and Yihuai Gao and Zipeng Fu and Jie Tan and Shuran Song},
  year={2024},
  booktitle={Proceedings of the 2024 Conference on Robot Learning},
}
```
