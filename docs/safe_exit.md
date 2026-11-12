# Failure exit: damping until `h`, then home

Every flash_aloha workflow that moves a real arm has the same exit
for failures. It is implemented once in `src/hardware/safe_exit.py`.

## What triggers it

Any catchable failure while an arm is under control:

- Python exceptions: joint-limit or target validation, camera or capture timing,
  policy/server errors, SDK errors raised to Python, stale feedback detected in
  Python, cross-arm feedback skew;
- Ctrl+C or SIGTERM;
- the shared runtime's command watchdog expiring (no new target for
  `command_timeout_s`).

Intentional quits (`x`/`q` at a prompt, a finished `--duration`) are not
failures and keep each tool's normal exit.

## What happens

1. **Damping**: the SDK's `set_to_damping()` (zero position gain, the SDK's
   default damping gains toward zero velocity; the SDK applies gravity
   compensation). Each gripper is held **where it is** with its default gains;
   it is never driven closed.
   - In the shared runtime the watchdog itself puts an expired arm into damping;
     the native SDK loop keeps sending that command even if Python stalls.
2. The arm stays damped until the operator presses **`h`** (terminal, the
   collector's key watcher, or the setup tool's camera window).
3. **Home**: the SDK's `reset_to_home()`, which ramps to its default gains
   itself and opens the gripper fully.
4. A failure during the home move returns to damping (step 1).
5. The tool then closes its controllers and exits with the original error.

**Ctrl+C while damped disables the arm in place.** This is the escape hatch:
hold the arm first.

## What cannot damp

Damping needs a working control loop. These still leave motors disabled:

- an ARX SDK that has itself entered its safety shutdown;
- SIGKILL, a native crash, or power loss.

With two arms, an arm that cannot damp is disabled while the other one keeps
damping and homes normally.

## Where it applies

| Workflow | Home after `h` |
|---|---|
| `record_episode_single.py`, including review replay and `--record-replay` | `reset_to_home()` |
| `visualize_episode.py --review_replay` (the collector's review replay) | `reset_to_home()` |
| `visualize_episode.py --action_replay` | `reset_to_home()` |
| `inference_direct.py`, `inference_remote.py` (legacy rollouts) | `reset_to_home()` |
| `record_hardware.py`, `inference_remote.py --shared-hardware` | the ARX adapter's native home (`reset_to_home()`) |
| `scripts/practice.py`, `set_up_recording.py`, `hold_start_position.py`, `calibrate_arx_arm.py`, `calibrate_masking.py`, `tune_masking.py --live` | `reset_to_home()` |

The collector's replay `1` stop key is a deliberate stop, not a failure: it
damps and `h` returns to the start position for the review.
