# Configuration

Task YAML (`tasks/`) defines cameras, physical arm/CAN assignments, capture/action
rate, prediction horizon and dataset location. Training recipes (`training/`)
select the policy architecture and training settings. Inference profiles (`inference/`)
select execution windows, smoothing, RTC and rollout recording.

Create validated configurations without starting hardware:

```bash
python scripts/config/create_config.py task my_pick --template single_arm --dry-run
python scripts/config/create_config.py training my_pick_act --dry-run
python scripts/config/create_config.py training my_pick_flow --model-type Flow --dry-run
python scripts/config/check_training_config.py --training-config single_arm_act
python scripts/config/check_inference_config.py single_arm_remote --schema-only
```

ACT is the default policy; Flow is also available. See `training/README.md` and
`inference/README.md` for the included recipes and execution profiles. Use explicit
YAML paths for customized tasks/recipes.
Task invariants cannot be overridden by inference profiles. Runtime RTC must
match the checkpoint's explicit training-time RTC declarations and delay range.

`scripts/run.py` runs every workflow by task or training-config name. It
resolves the paths and arguments, then starts one of the three launchers
`scripts/record.py`, `scripts/infer.py` and `scripts/visualize.py` (or the
training launcher), which run the unchanged `src/` entry points:

| Command | Runs |
| --- | --- |
| `run.py collect TASK` | `record.py single` |
| `run.py train TRAINING` | `scripts/policy/imitate_episodes_train.py` |
| `run.py infer TRAINING` | `infer.py direct` with the checkpoint and same-named profile |
| `run.py infer TRAINING --inference-config REMOTE_PROFILE` | `infer.py remote` (profile `backend: remote`) |
| `run.py serve TRAINING` | `infer.py serve` with the checkpoint |
| `run.py view TASK EPISODE` | `visualize.py episode --rerun` |
| `run.py replay TASK EPISODE [EPISODE ...]` | `visualize.py episode --review_replay` (the collector's review replay) |

Options a workflow does not define are passed unchanged to its entry point,
e.g. `run.py serve single_arm_act --device cuda:0` or
`run.py infer single_arm_flow --inference-config single_arm_remote --remote_host HOST`.
`--dry-run` prints the resolved command as JSON without starting anything:

```bash
python scripts/run.py collect single_arm --dry-run
python scripts/run.py train single_arm_act --dry-run
python scripts/run.py infer single_arm_act --dry-run
python scripts/run.py replay single_arm 24 27 --dry-run
```

Collection, inference, setup and replay are operator-run. A remote client
cannot select the checkpoint step; the server owns model selection
(`run.py serve TRAINING --policy-step N`).
