# Execution profiles

`single_arm_act.yaml` and `bimanual_act.yaml` select direct ACT inference with
sigmoid chunk blending. `single_arm_flow.yaml` selects direct Flow inference with
sigmoid chunk blending. `single_arm_remote.yaml` selects asynchronous remote
inference with RTC against this repository's model server, and
`single_arm_openpi_remote.yaml` against an OpenPI server; see
`docs/image_preprocessing.md` for the contract that server must advertise.

Setting `rtc.enabled: true` (with `rtc.training_time: true`) on a direct profile selects direct inference with RTC: each chunk after the
first is conditioned on the previous plan's next actions, as a prefix pinned at
clean time. The prefix length is the checkpoint's `training_time_rtc_max_delay`
(not configurable here). Direct RTC is always asynchronous, like remote mode, on the
same query worker: each query is submitted the prefix length before its chunk
boundary (15 steps for a 15-action delay) with the plan's next actions pinned, the
robot keeps executing them while the policy runs, and at the boundary the new chunk
takes over from index 15. No planned action is skipped. A response that is not ready
by the boundary, or took the whole overlap, fails closed; so does a warm-up query
with a prefix that is slower than the overlap. `action_lookahead` does not apply and
is rejected with RTC; the prefix must be shorter than `execution.chunk_steps` and
`execution.first_chunk_steps`. As remote mode resets its session, the first query of
an episode, after a ready reset, after continuing from a pause and when policy
control resumes after a DAgger hand correction is a fresh synchronous query without
a prefix that restarts the chunk schedule; a query still running is discarded.
Blending under RTC crossfades only the new chunk's side of the handoff:
the `blending.steps` executed actions after it fade from the previous plan into the
new chunk's predictions. It is applied after the prediction and never changes the
policy's input or the next prefix, which comes from the unblended prediction.
The previous plan must still hold the `blending.steps` actions after the handoff
(chunk steps + prefix + `blending.steps` within the prediction horizon), the
prefix plus `blending.steps` must fit in `execution.chunk_steps`, and there is no
blending after a fresh start. Without RTC, `blending.steps` keeps its
chunk-blending meaning (the number of blended actions).
nn_chunk_search and temporal aggregation are rejected.
A checkpoint trained with `data.masking_strategy` runs through direct inference only,
which applies its mask; the model server and the remote client refuse it.
Validate with `python scripts/config/check_inference_config.py NAME --schema-only`;
omit `--schema-only` to validate against checkpoint metadata (remote profiles
also require `--checkpoint-config`). Rates, cameras, dimensions and physical
setup remain task/checkpoint invariants, not execution-profile overrides.

`options.fast_inference: true` (direct; `--fast_inference` for `serve_policy`)
CUDA-graphs the batch-1 ACT forward (channels-last backbones), and speeds up Flow sampling on CUDA: the U-Net's Conv1d layers run as channels-last
Conv2d with fp16 weights stored once (cuDNN otherwise reorders the weights on
every call), and each U-Net step is replayed from a CUDA graph. The checkpoint,
objective and sampler are unchanged; actions differ from fp32 by about 5e-4 rad.
On an RTX 3070 Ti laptop GPU a Flow query takes ~17 ms instead of ~70 ms.
