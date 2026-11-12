"""scripts/run.py plans and the scripts/ mode launchers, without starting any process."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

import _launcher  # noqa: E402
import infer  # noqa: E402
import record  # noqa: E402
import tune  # noqa: E402
import visualize  # noqa: E402
from config.run import build_parser, build_plan  # noqa: E402


@pytest.mark.parametrize('launcher', [infer, record, tune, visualize])
def test_every_mode_runs_an_existing_src_entry_point(launcher):
    for mode, (path, _) in launcher.MODES.items():
        assert (ROOT / path).is_file(), mode
        command = _launcher.command(launcher.MODES, mode, ['--x', '1'])
        assert command[-3:] == [str(ROOT / path), '--x', '1']  # arguments pass through unchanged


def test_unknown_or_missing_mode_prints_usage_and_fails(capsys):
    with pytest.raises(SystemExit) as missing:
        _launcher.main('summary', infer.MODES, [])
    assert missing.value.code == 2 and 'direct' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="unknown mode 'bogus'"):
        _launcher.main('summary', infer.MODES, ['bogus'])


def plan(*argv):
    args, extra = build_parser().parse_known_args(argv)
    command = build_plan(args, extra)['command']
    assert command[0] == sys.executable
    return [Path(part).name if part.startswith(str(ROOT)) else part for part in command[1:]]


def test_collect_and_train_use_the_launchers():
    assert plan('collect', 'arx_left_wrist_only')[:2] == ['record.py', 'single']
    assert plan('train', 'single_arm_flow') == [
        'imitate_episodes_train.py', '--training-config', 'single_arm_flow.yaml']


def test_infer_selects_direct_or_remote_by_profile_backend():
    direct = plan('infer', 'single_arm_flow', '--policy-step', '7')
    assert direct[:2] == ['infer.py', 'direct']
    assert Path(direct[direct.index('--config') + 1]).name == 'all_configs.json'  # ckpts may be a symlink
    assert direct[-2:] == ['--policy_step', '7']
    remote = plan('infer', 'single_arm_flow', '--inference-config', 'single_arm_remote',
                  '--remote_host', 'gpu-box')
    assert remote[:2] == ['infer.py', 'remote'] and '--config' not in remote
    assert remote[-2:] == ['--remote_host', 'gpu-box']  # undeclared options pass through
    with pytest.raises(ValueError, match='checkpoint step'):
        plan('infer', 'single_arm_flow', '--inference-config', 'single_arm_remote', '--policy-step', '5')


def test_serve_passes_checkpoint_and_extra_options():
    command = plan('serve', 'single_arm_flow', '--device', 'cuda:0')
    assert command[:2] == ['infer.py', 'serve']
    assert Path(command[command.index('--config') + 1]).name == 'all_configs.json'
    assert command[-2:] == ['--device', 'cuda:0']


def test_view_and_replay_resolve_robot_from_the_task():
    view = plan('view', 'arx_left_wrist_only', '24')
    assert view[:4] == ['visualize.py', 'episode', '--robot', 'arx']
    assert view[-3:] == ['--episode_idx', '24', '--rerun']
    assert plan('replay', 'arx_left_wrist_only', '24', '27')[-3:] == ['--review_replay', '24', '27']
    assert plan('replay', 'arx_right', '3')[-2:] == ['--review_replay', '3']


def test_tuning_is_gravity_then_friction_on_the_arx_tuners():
    assert list(tune.MODES) == ['gravity', 'friction']
    assert all((ROOT / path).is_file() for path, _ in tune.MODES.values())
    arx = tune.command(['gravity', 'L5', 'can0'])
    assert arx[-3:] == [str(ROOT / 'src/hardware/arx5-sdk/python/examples/tune_gravity_compensation.py'), 'L5', 'can0']
    friction = tune.command(['friction', 'L5', 'can1'])
    assert friction[-3:] == [str(ROOT / 'src/hardware/arx5-sdk/python/examples/tune_gripper_friction_compensation.py'), 'L5', 'can1']
