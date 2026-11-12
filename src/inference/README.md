# Inference

- `inference_direct.py`: loads an ACT or Flow checkpoint locally and runs the robot; chunk blending, lookahead and direct RTC.
- `inference_remote.py`: robot-side WebSocket client of a policy server; asynchronous fixed-overlap RTC, recording and generic DAgger.
- `serve_policy.py`: model server for ACT and Flow checkpoints with NumPy/MessagePack transport.
- `policy_chunks.py`: chunk scheduling, overlap checks and RTC prefix handling.
- `policy_transport.py`: transport and timing helpers.

The operator entry point is `scripts/infer.py` (modes `direct`, `remote`, `serve`). See the repository README
for complete commands and the differences between direct and remote
execution. Both modes reuse the capture, image, action and recording contracts.
The remote client also talks to an OpenPI server (for example Pi0.5) that
advertises the client image contract; see `docs/image_preprocessing.md`.
