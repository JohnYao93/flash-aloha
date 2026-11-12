# Standard image processing

`src/contracts/standard_image_processing.py` owns deterministic image preprocessing. Both
training and deployment must call it, using the same profile and dimensions.
The observation capture contract selects frames and proprio together; this
separate contract governs what happens to those selected pixels.

Custom processing must build on `standard_image_processing`: compose its
operations into a named policy-specific profile, reuse that profile on both
sides, and add a parity test. Do not copy the implementation into a custom
trainer or runner. Different model families may require different profiles;
for example, ACT's CHW [0, 1] output is not OpenPI's final model input format.

## Profiles and callers

| Profile | Training | Deployment |
| --- | --- | --- |
| Standard ACT/Flow | `training_utils.EpisodicDataset` | direct inference and `serve_policy.LocalPolicyServer` |
| OpenPI physical cameras (`openpi_letterbox_448`) | native RGB export, then OpenPI `resize_with_pad` to 448x448 | client letterboxes to 448x448 RGB (`openpi_request_images`); server checks, never resizes |

Standard images use OpenCV INTER_AREA, explicit BGR/RGB conversion, float32
division by 255, CHW layout, and optional mean-channel grayscale. ALOHA's legacy
BGR model convention is retained when its checkpoint/task configuration enables
`aloha`. The websocket server rejects masked and image-history checkpoints.
Custom height/width come from the checkpoint configuration. Conflicting
top-level and `policy_config` dimensions fail rather than selecting different
sizes for direct versus served inference.

### OpenPI: client-side letterbox

The remote `inference_remote` client letterboxes each native camera frame itself and
sends 448x448 HWC RGB uint8, about 4.6x fewer bytes than two native 1280x720 frames.
The profile `openpi_letterbox_448` composes the standard operations: BGR->RGB, an
INTER_LINEAR resize to fit (`letterbox_image`, 1280x720 -> 448x252), centered on a
zero canvas (content rows 98..349). Its geometry, float truncation included, is that
of `openpi_client.image_tools.resize_with_pad`, which OpenPI training applies to
native frames. No resize is baked into recordings; training data for OpenPI must keep
native pixels (BGR->RGB only).

The contract is declared once, here (`openpi_image_contract`). This repository does
not export OpenPI training data or write the contract file; the OpenPI side must
produce and advertise it: the profile, camera request keys and source shapes, model
input shape, and a pixel fingerprint, plus the policy metadata the client requires
(task, rate, dimensions, robot and camera-settings contracts). The fingerprint is the
SHA-256 of the letterboxed `xorhash-v1` probe image (`letterbox_probe`, integer
arithmetic only) for every source shape, so it compares actual pixels across OpenCV
builds instead of version strings.

The OpenPI side must integrate the same contract before it can serve this client:
training recomputes the fingerprint with its own resize and refuses to start on a
mismatch, every checkpoint stores the contract, and the server advertises it in its
metadata (`image_preprocessing`, with `model_family: openpi` and
`request_schema: openpi`), checks for 448x448 uint8 inputs instead of resizing, and
rejects any request whose `client/image_preprocessing_sha256` differs. At startup the
client rebuilds the contract with its own OpenCV and requires exact equality,
including the camera order. A server that does not advertise the contract is refused.

## Masking

`src/utils/masking.py` (`Masker`) hides parts of the policy's inputs. Masking is defined
by data: the masking config
([`calibration/masking/masking.yaml`](../calibration/masking/README.md), or the file named
by `FLASH_ALOHA_MASKING_CONFIG`) declares the calibration of one physical setup and every
strategy as a list of operations: polygons in metres that follow an arm link or sit at
its base and are projected into the overhead or wrist cameras, image regions given as
fractions, blanked cameras, and zeroed proprioception. Add or change a strategy by
editing that file; no code changes.

A training recipe selects a strategy with `data.masking_strategy`. The checkpoint stores
it together with a contract of the strategy's operations and calibration, and direct
inference applies the same `Masker` to every policy observation after checking that
contract (`--masking_strategy` may confirm the strategy but cannot change it). The model
server and the remote client do not apply masks and refuse masked checkpoints. Recordings
are never masked.

The mask runs on the policy's images, after the standard resize and before
channel/scaling conversion, and on the raw qpos, in the same order in training and
deployment. Each observation's mask is computed from that observation's own qpos. Masks
are resolution independent: intrinsics are rescaled from the calibrated image size to the
size of the image being masked.

The shipped strategies: `end_effector` (overhead: a rectangle that follows each arm's
wrist; wrist cameras: rectangles that follow the other arm's end effector),
`moving_block` (wrist cameras: a wall in front of the arm base; overhead: the top
quarter), `arm_cameras` (wrist images zeroed), `overhead_left_90px` and
`left_right_overhead` (fixed columns of the overhead image), `prioproceptive` (qpos
zeroed and not normalized), and the combinations `end_effector_and_prioproceptive`,
`prioproceptive_and_moving_block` and `arm_cameras_and_prioproceptive`.

Projected strategies need the state of every arm the masking config declares and that
setup's calibration, which `scripts/setup/calibrate_masking.py` and
`scripts/cameras/camera_int.py` produce. To check a calibration on recorded data, open an
episode with
`python scripts/visualize.py episode --data-dir EPISODE --training-config RECIPE`: the
preview shows each camera at the policy resolution with the recipe's mask applied to
every frame from that frame's recorded qpos. To adjust the polygons interactively
(optional), use `scripts/setup/tune_masking.py`. Run `scripts/tests/test_masking.py` and
`scripts/tests/test_masking_calibration.py` when changing either.

## Augmentations and storage

- Stochastic color/spatial augmentation is training-only and happens after the
  shared deterministic stage.
- Backbone normalization stays inside the shared policy/model, not an extra
  inference-only image transform.
- Recordings stay native; no preprocessing is baked into collection.
- Equality means identical decoded inputs and configuration produce identical
  preprocessed tensors. Lossy video/JPEG storage, optional inference JPEG
  simulation, and different decoders can change the input pixels themselves.
- Do not infer parity for external training from dimensions alone. Audit the
  deterministic profile; keep random training augmentation separate and disabled
  during the comparison.

Run `scripts/tests/test_standard_image_processing.py` plus the ACT, Flow and
standard-capture regression suites when modifying either side. These tests use
synthetic images/Zarr episode data and do not access cameras or robots. Set
`OPENPI_CLIENT_SRC` to an OpenPI checkout's `packages/openpi-client/src` to also
compare the letterbox against OpenPI's own `resize_with_pad`.
