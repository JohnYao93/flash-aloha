# Hand-guided recording (operator-run)

```bash
bash scripts/cameras/camera_servers_tmux.sh start left overhead
uv run scripts/record.py single --task_name single_arm
```

Start the camera server only if it is not already running. The recorder is
`ARXBackend` in `src/recording/record_episode_single.py`; it drives one or two
ARX L5 arms. Takes are started and stopped with the `1` key or the left pedal
only.

In **motion mode**, the recorder moves to the task's start pose (see
[saved start poses](arm_bringup.md#saving-start-poses)), including the saved
gripper value, and holds all six joints and the gripper with the collection
tracking gains. Each take reloads the start-pose file and prints its gripper
target. Add `--no-start-position` to hold the arm where it is instead of moving.
Press `1` to enter **recording mode**: the arms and grippers go passive
(kp = kd = 0 with the SDK's gravity and friction feedforward; friction
feedforward is off whenever the arm holds, moves or replays). The start-pose setup
tool and DAgger corrections during inference follow the same rule: friction
feedforward is on only while the operator is moving the arm or gripper by hand. Move the arm by
hand and press `1` to stop early (otherwise the task duration ends the take).
The recorder returns to motion mode, holding the measured arm and gripper pose
for review. Press `1` to save, `h`/`n` to discard, `r` to replay the unsaved
take before deciding (see [review replay](recording_review_replay.md)), or `q`
to exit. These keys match the two-pedal foot switch (left `1`, right `h`; see
`src/recording/pedal_keys.py`). While recording, `f` stamps a failure event,
`2` a subgoal switch, and `x` aborts the take. After a decision, press Enter to
prepare another take, `r` to replay the last saved episode, or `x` to quit.
Episodes auto-increment under `data/single_arm/`.

Recorded positions are the SDK's joint radians and the gripper width in metres.
Quitting with `x`/`q` homes each arm with the SDK's `reset_to_home()`, which
opens the gripper, and then closes the controllers. Ctrl+C, SIGTERM and Python
errors enter damping until `h` homes the arm ([docs/safe_exit.md](safe_exit.md)).
SIGKILL, native crashes or power loss cannot damp.

## Guided start

`--guided-start` lets a take start anywhere. Each take skips the saved-pose move
and goes into **guided mode** instead: the same passive support as recording
mode, with nothing captured. Move the arm by hand, then press `1` to **freeze**
it there. This is the same measured arm and gripper hold used after a take.
Press `1` again to record from the frozen pose, exactly as in a normal take.
While frozen, `h`/`n` (right pedal) releases the arm so you can guide it again.
`x` holds the arm, then quits through the normal home move. Episodes record
`guided_start: true` and `start_position_applied: false`; the first recorded
row is the start pose. A review replay that is stopped and returned with `h`
still goes to the saved start pose.

```bash
uv run scripts/run.py collect TASK --guided-start
```

## Recording a replay of each take

Add `--record-replay` (auto-incremented episodes only) to also record a replay
of each saved take. After the save, `1` moves to the take's first pose and
shows the alignment overlay; reset the scene and press `1` to replay it while
recording (`1` stops, and a stopped replay is not saved). The replay uses this
session's kp/kd fractions without prompting (`r` at the next-episode prompt
changes them). Review it like a take: `1` saves it as the next episode, `h`/`n`
discards. Its actions are the replayed rows and its observations are measured
during the replay; metadata `replay_recording: true`, `replay_source_episode`,
`replay_kp_scale` and `replay_kd_scale` mark it. At the prompt after the save,
Enter prepares another take without recording a replay.
