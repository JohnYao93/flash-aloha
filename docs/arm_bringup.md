# Connecting and checking an arm (operator-run)

Step-by-step bring-up for ARX arms: identifying the USB-CAN adapter, activating the
CAN interface, selecting the arm in a task YAML, checking the SDK, tuning compensation
and saving start poses. Run these commands yourself, with arm controllers stopped
before reconfiguring CAN. The repository README has the short version.

## Identify the USB-CAN adapter

Compare the devices before and after connecting each adapter:

```bash
ip -br link
ls -l /dev/serial/by-id/
```

| Adapter | What appears | Bring-up method |
| --- | --- | --- |
| candleLight USB-CAN | A native CAN interface, commonly `can0` | Native SocketCAN commands below |
| SLCAN USB-CAN | A serial device such as `/dev/ttyACM0` | `slcand` commands below |

The adapter determines the bring-up method. Interface numbering may change after
reconnecting adapters. Identify each physical arm's adapter before assigning
`can0`/`can1` to arm roles. EtherCAT-CAN setups use the separate procedure in the
[ARX SDK README](../src/hardware/arx5-sdk/README.md#ethercat-can-setup).

## Activate native SocketCAN (candleLight)

Once the intended adapter is identified as `can0`, configure it at **1 Mbit/s**:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details -statistics link show can0
```

Check that the interface flags include `UP`, the bitrate is `1000000`, and the
CAN state is not `BUS-OFF`. Repeat after a reboot or adapter reconnect unless
the host already configures the interface automatically. For a second arm,
repeat with its actual interface name, such as `can1`. If `can0` does not exist,
resolve adapter detection first; `ip link set` configures an existing interface.
For `BUS-OFF`, check wiring, termination, power and bitrate before restarting.

This follows the [ARX USB-CAN setup](../src/hardware/arx5-sdk/README.md#usb-can-setup) and
[Linux SocketCAN documentation](https://docs.kernel.org/networking/can.html).

## Activate ARX SLCAN

Install the serial CAN utility once if needed:

```bash
sudo apt-get install can-utils
```

For an adapter already registered as `/dev/arxcan0`:

```bash
sudo slcand -o -f -s8 /dev/arxcan0 can0
sudo ip link set can0 up
ip -details -statistics link show can0
```

`-s8` selects 1 Mbit/s. For a new adapter without that symlink, replace
`/dev/arxcan0` with its identified serial device, for example `/dev/ttyACM0`.
Configure a second adapter with its own device path and an unused interface
name such as `can1`; do not launch a duplicate `slcand` for an attached adapter.
SLCAN's bitrate is set by `slcand`, rather than the native SocketCAN bitrate
command. See the [ARX SDK instructions](../src/hardware/arx5-sdk/README.md#for-adapters-using-slcan-framework)
for registering adapter serial numbers with persistent device names.

## ARX utility scripts

These shortcuts wrap the ARX setup steps above. The
[ARX SDK README](../src/hardware/arx5-sdk/README.md) explains the underlying
adapter setup and controller gains.

| Script | What it does |
| --- | --- |
| `source arx_setup.sh` | Activates `.venv` and sources `scripts/setup/paths.sh`; run from the repository root |
| `bash scripts/setup/arx_arms.sh` | Brings up both SLCAN adapters (`/dev/arxcan0` as `can0`, `/dev/arxcan1` as `can1`) |
| `scripts/setup/activate_arm 0` (or `1`) | Brings native SocketCAN `can0` (or `can1`) up at 1 Mbit/s and prints its status; run with that arm's controller stopped |
| `python src/hardware/check_gain.py` | Prints the ARX SDK's default joint and gripper gains for the L5, with example scaled values; it does not connect to an arm |

## Select the arm in a task YAML and check the SDK

Match each task YAML's `robot.arms.<left|right>.can_interface` to the actual
adapter. Tasks use `robot.type: arx`; each arm may name its `model` (`L5`, the
default, or `Trossen_L5`) and may set `command_timeout_s` and
`max_feedback_age_s`. Start from
[single_arm.yaml](../src/config/tasks/single_arm.yaml) (one arm, two cameras) or
[bimanual.yaml](../src/config/tasks/bimanual.yaml) (two arms, three cameras); every
field is described in the [task README](../src/config/tasks/README.md). Set
`FLASH_ALOHA_TASK_DIRS` to keep your own task YAMLs outside the repository and
still select them by name. The files under `calibration/` are examples from one
setup; new physical arms need their own tuning and start poses.

Verify the built SDK extension is importable in the root environment:

```bash
python -c "import arx5_interface; print(arx5_interface.__file__)"
```

If a controller then fails to start or reports no feedback, inspect
`ip -details -statistics link show can0`: an interface that is `UP` at 1 Mbit/s
with zero received packets has not received motor replies. Verify that the arm
has motor power, the adapter is connected to that physical arm, and
wiring/termination are correct. Use only one controller per CAN interface.

## Tuning details

`uv run scripts/tune.py gravity MODEL INTERFACE` and `friction MODEL INTERFACE`
run the ARX SDK tuners
(`src/hardware/arx5-sdk/python/examples/tune_gravity_compensation.py` and
`tune_gripper_friction_compensation.py`) with their arguments unchanged; `MODEL`
is `L5` or `Trossen_L5`. Tune gravity first, then friction, once per physical arm:

```bash
uv run scripts/tune.py gravity L5 can0
uv run scripts/tune.py friction L5 can0
```

Both tuners enable the motors with zero position gain, and exit puts the arm into
SDK damping without a home move. Support the arm before starting and before
exiting, and use only one controller on the CAN interface. Each tuner shows its
keys on the second line of its terminal UI:

| Tuner | Keys |
| --- | --- |
| gravity | `[` / `]` select a link, up/down change its mass, Space toggles gravity compensation, `+` / `-` change the step, `r` / `R` reset the link / all links, `p` pauses, `l` loads the saved values, `s` saves, `q` quits |
| friction | `[` / `]` select the gripper or a joint, arrows select and change a term, Space toggles all friction compensation, `B` / `G` toggle joint (body) / gripper friction, `b` / `v` apply a small torque bump in either direction, `g` toggles gravity compensation, `0` zeroes the field, `r` / `R` reset, `+` / `-` change the step, `p` pauses, `l` loads the saved values, `s` saves, `q` quits |

`s` writes the link masses, or the friction terms, into the selected model's URDF
in the SDK (`--urdf-path` selects another file). Restart a running script after
tuning to load the new values. `scripts/tune.py gravity --help` and
`scripts/tune.py friction --help` list the remaining options.

## Saving start poses

`uv run scripts/setup/set_up_recording.py --task_name TASK` saves
`calibration/start_positions/TASK.npz` (see the README). Without a task, select
the hardware directly; the pose is then saved to the shared
`calibration/start_position.npz`:

```bash
uv run scripts/setup/set_up_recording.py --arm right --can-interface can1
```

For two arms, use `--arm both`: left defaults to `can0` and right to `can1`.
Use `--can-interface` (or `--left-can-interface` / `--right-can-interface`) only
to override those defaults. The camera names follow `--arm`; override them with
`--cameras`. `--cameras none` opens no cameras; type Space/`y`/`n`/`q` (and `h`
in damping) in the terminal instead of a camera window.

The tool starts in practice mode: arm and gripper gains are zero and the SDK
provides gravity compensation, so the arm and gripper move by hand. Space locks
the measured arm and gripper pose with the collection gains; `n` returns to
practice, `y` saves and `q` exits. `--gripper-only` instead moves to the
existing saved pose (over `--move-duration` seconds), holds the six arm joints
and lets you adjust only the gripper before saving; `--source-position` names
the pose to start from. `--reset-home` homes the arm(s) first. A normal exit
returns the arm(s) home with the SDK's `reset_to_home()`; Ctrl+C or a failure
enters damping until `h` homes the arm ([docs/safe_exit.md](safe_exit.md)).
Support the arm during startup and shutdown.

Saved positions are SDK joint radians, with the gripper as a width in metres.
Use `--output` for another destination. The recorder and inference start from
the saved pose including its gripper value, so resave it after changing how
the gripper should start.
