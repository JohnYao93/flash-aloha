# Saved camera settings contract

Episodes carry an optional `camera_settings` mapping alongside their rate and
observation metadata. It records the tuning values saved by
`scripts/cameras/tune_cameras.py`, not a live query of the camera hardware.

Collection automatically snapshots the file at episode start. It uses the camera
server's path resolution: `FLASH_ALOHA_CAMERA_PARAMS` when set, otherwise
`calibration/camera_params.npz`, falling back to `camera_params.npz` at the repository
root. Use the environment variable when tuning with a custom `--output` path.
Only the episode's cameras are included. Camera aliases are resolved by the shared
camera-server loader. The snapshot contains controls (exposure, gain, white balance,
etc.) and the saved `[width, height, camera_fps]` mode. Device paths, file paths,
file modification times, NPZ ordering and unused cameras are not compared.

Raw episode manifests store the mapping under `metadata.camera_settings`.
Compatibility Zarr episode exports store the same mapping as JSON in the `camera_settings`
attribute. Export uses the start snapshot; it does not reread the tuning file at
save time. Recorded pixels, frame IDs, timestamps, image processing and capture
ordering are unchanged.

Standard ACT/Flow training combines the known settings from its selected episodes
and writes `camera_settings` into checkpoint configuration. Contradictory values
for the same camera/control or mode fail training. Episodes with empty or missing
settings add no constraint. Local policy serving passes the checkpoint mapping
through its server metadata. Direct inference and websocket inference compare it
against the local NPZ before robot setup, including alternate policy endpoints.
External trainers/servers must propagate this field to enable the same check;
older checkpoints/servers without it remain usable.

## Optional values

- Missing metadata, `null`, an empty string, or `{}` means no constraint.
- An explicitly empty top-level checkpoint/server mapping disables matching of
  settings declared only in nested checkpoint sections.
- A missing tuning file or camera record is allowed. Missing controls/modes add
  no constraint; when both sides declare a value, they must agree.
- A malformed existing NPZ or invalid nonempty contract is an error, rather than
  being treated as an absent declaration.
- A filename match is insufficient: changing exposure or saved resolution/FPS
  changes the compared values even when the NPZ path stays the same.

An optional task YAML mapping can add a required setting, for example:

```yaml
camera_settings:
  video_overhead:
    settings:
      exposure_absolute: 156
      gain: 30
    mode: [1280, 800, 100]
```

The default task value is `camera_settings: {}`: the task adds no constraint.
Collection still records an available tuning snapshot, and a nonempty checkpoint
snapshot is still checked at inference. Task compatibility checks compare known
episode/checkpoint declarations with any nonempty task requirement. The entire
`camera_settings` mapping is replaced by an explicit YAML override, so `{}` can
clear a task requirement. A deliberately unrestricted checkpoint can likewise set
its top-level `camera_settings` to `{}`; a separately declared nonempty task
requirement still applies.

The mode's camera FPS is separate from episode/action FPS. This contract checks
the saved tuning artifact requested for collection/deployment; it does not verify
that an already-running camera server has reloaded a changed file or that live
hardware controls match it.
