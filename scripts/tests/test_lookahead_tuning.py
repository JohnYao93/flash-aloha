"""Pause-time tuning of the direct-inference action and gripper lookahead (no robot)."""
from inference.lookahead_tuning import LookaheadTuner


def tuner(action=10, gripper=5, **kwargs):
    log = []
    options = dict(chunk_steps=50, horizon=100, fps=100, notify=log.append)
    return LookaheadTuner(action, gripper, **{**options, **kwargs}), log


def test_keys_step_the_two_numbers_and_report_milliseconds():
    t, log = tuner()
    for key in ('w', '\x1b[a', 's', 'd', '\x1b[c', '\x1b[d', 'a', 'a'):
        assert t.adjust(key)
    assert (t.action, t.gripper) == (11, 4)
    assert log[-1].startswith('[Lookahead] action 11 (110 ms), gripper +4 (150 ms)')
    assert not t.adjust('c') and not t.adjust('1') and len(log) == 8


def test_values_that_run_out_of_the_chunk_or_below_zero_are_refused():
    t, log = tuner(action=40, gripper=10)                 # 50 + 40 + 10 == horizon
    assert t.adjust('w') and t.adjust('d')
    assert (t.action, t.gripper) == (40, 10) and all('refused' in line for line in log)
    t, log = tuner(action=0, gripper=0)
    assert t.adjust('s') and t.adjust('a')
    assert (t.action, t.gripper) == (0, 0) and all('below 0' in line for line in log)


def test_inference_loop_reads_the_tuner_only_when_paused_and_never_skips_a_first_chunk():
    from pathlib import Path
    import inference.inference_direct as direct
    source = Path(direct.__file__).read_text()
    assert 'if pause_flag and lookahead_tuner is not None and lookahead_tuner.adjust(key):' in source
    assert source.count('current_chunk_start_index = 0\n') == 2
    assert 'if rtc_prefix_steps else action_lookahead' not in source
    assert direct.lookahead_tuner is None                 # nothing to tune until a rollout starts
