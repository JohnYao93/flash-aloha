# Policies and training

`act.py` contains `ACTPolicy`, a decoder-only ACT and the default policy.
`policy.py` contains `LeRobotFlowPolicy` (Flow matching on the LeRobot U-Net and
visual backbone). `model_util.py` accepts `ACT` and `Flow`. Both support
optional training-time RTC prefix conditioning.

Train with `scripts/policy/imitate_episodes_train.py --training-config single_arm_act`
(or `bimanual_act`, or a `*_flow` recipe). Image transforms reuse
`contracts/standard_image_processing.py`. Raw data readers preserve saved
references, timestamps and continuity boundaries.
