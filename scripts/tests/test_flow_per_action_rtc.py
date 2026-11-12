"""The per-action-time Flow objective: offline, CPU only, no robot."""
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

import pytest
import torch

from policy.flow_rtc import PerActionTimeUNet, ContinuousTimeEmbedding, per_action_flow_loss, per_action_sample
from policy.policy import LeRobotFlowPolicy


def test_prefix_is_clean_at_time_zero_and_loss_is_masked_velocity_mse():
    actions = torch.full((2, 4, 7), 3.0)
    noise = torch.ones_like(actions)
    prediction = torch.zeros_like(actions, requires_grad=True)
    seen = []

    def predict(x, time):
        seen.append((x.detach().clone(), time.clone()))
        return prediction

    padding = torch.tensor([[False, False, False, True], [False, False, False, False]])
    output = per_action_flow_loss(actions, noise, torch.tensor([0.5, 0.25]), torch.tensor([1, 2]), predict, padding)
    assert output['loss'].item() == 4.0  # squared error of the -2 target velocity (noise - actions)
    x, time = seen[0]
    torch.testing.assert_close(time, torch.tensor([[0, .5, .5, .5], [0, 0, .25, .25]]))
    torch.testing.assert_close(x[0, :1], actions[0, :1])  # prefix: exactly the clean actions
    torch.testing.assert_close(x[1, :2], actions[1, :2])
    torch.testing.assert_close(x[0, 1:], torch.full((3, 7), 2.0))  # t * noise + (1 - t) * actions
    torch.testing.assert_close(x[1, 2:], torch.full((2, 7), 2.5))
    output['loss'].backward()
    assert torch.count_nonzero(prediction.grad[0, 0]) == 0  # prefix unsupervised
    assert torch.count_nonzero(prediction.grad[0, 3]) == 0  # padding unsupervised
    assert torch.count_nonzero(prediction.grad[1, :2]) == 0
    assert torch.all(prediction.grad[0, 1] > 0) and torch.all(prediction.grad[1, 2] > 0)


def test_every_example_has_equal_weight_whatever_its_delay():
    actions = torch.zeros((2, 4, 1))
    prediction = torch.tensor([[[2.0]] * 4, [[4.0]] * 4])
    output = per_action_flow_loss(actions, actions, torch.ones(2), torch.tensor([0, 3]), lambda x, t: prediction)
    assert output['loss'].item() == 10.0  # (4 + 16) / 2, not weighted by valid-action counts


def test_all_padding_gives_zero_loss_and_gradients():
    actions = torch.zeros((1, 4, 7))
    prediction = torch.ones_like(actions, requires_grad=True)
    output = per_action_flow_loss(actions, actions, torch.ones(1), torch.zeros(1, dtype=torch.long),
                            lambda x, t: prediction, torch.ones((1, 4), dtype=torch.bool))
    output['loss'].backward()
    assert output['loss'].item() == 0 and torch.count_nonzero(prediction.grad) == 0


def test_sampler_pins_the_prefix_at_time_zero_on_every_step_and_integrates_from_noise_to_data():
    calls = []

    def unet(x, t, condition):
        calls.append((x.clone(), t.clone()))
        return torch.ones_like(x)

    prefix = torch.arange(28.).reshape(1, 4, 7).expand(2, -1, -1)
    result = per_action_sample(unet, None, torch.zeros((2, 16, 7)), 8, prefix)
    assert len(calls) == 8
    for index, (x, time) in enumerate(calls):
        torch.testing.assert_close(x[:, :4], prefix)
        assert torch.count_nonzero(time[:, :4]) == 0
        torch.testing.assert_close(time[:, 4:], torch.full((2, 12), 1 - index / 8))
    torch.testing.assert_close(result[:, :4], prefix)
    torch.testing.assert_close(result[:, 4:], torch.full((2, 12, 7), -1.0))  # dx = (0 - 1) * velocity 1


def test_time_embedding_periods():
    embedded = ContinuousTimeEmbedding(8)(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(embedded[0], torch.tensor([0., 0, 0, 0, 1, 1, 1, 1]))
    periods = 0.004 * 1000.0 ** torch.linspace(0, 1, 4)
    torch.testing.assert_close(embedded[1, :4], torch.sin(2 * torch.pi / periods))


def small_policy(**override):
    config = dict(lr=1e-4, camera_names=['video_left'], height=24, width=32, state_dim=7, action_dim=7,
                  num_queries=16, hidden_dim=32, unet_depth=2, backbone='resnet18', training_time_rtc=True,
                  training_time_rtc_min_delay=0, training_time_rtc_max_delay=4, num_inference_timesteps=5)
    config.update(override)
    return LeRobotFlowPolicy(config)


def test_policy_uses_per_action_time_unet_and_its_actions_get_their_own_condition():
    torch.manual_seed(0)
    policy = small_policy()
    unet = policy.model.diffusion.unet
    assert isinstance(unet, PerActionTimeUNet)
    x, condition = torch.randn(1, 16, 7), torch.randn(1, unet.down_modules[0][0].cond_encoder[1].in_features - 128)
    same = unet(x, torch.full((1, 16), 0.7), condition)
    early = torch.full((1, 16), 0.7)
    early[:, :4] = 0
    changed = unet(x, early, condition)
    assert not torch.allclose(same, changed)
    with pytest.raises(ValueError, match='Per-action noise times'):
        unet(x, torch.full((1,), 0.7), condition)


def test_small_policy_trains_and_samples_with_a_trained_length_prefix():
    torch.manual_seed(0)
    policy = small_policy()
    qpos, image = torch.randn(3, 7), torch.rand(3, 1, 3, 24, 32)
    output = policy(qpos, image, torch.randn(3, 16, 7), torch.zeros(3, 16, dtype=torch.bool))
    output['loss'].backward()
    assert torch.isfinite(output['loss'])
    policy.eval()
    prefix = torch.randn(1, 4, 7)
    chunk = policy.forward_inf_training_time_rtc(qpos[:1], image[:1], prefix)
    assert chunk.shape == (1, 16, 7) and torch.equal(chunk[:, :4], prefix)
    assert policy.forward_inf(qpos[:1], image[:1]).shape == (1, 16, 7)
    for prefix in (torch.randn(1, 5, 7), torch.randn(1, 2, 6), torch.randn(2, 2, 7)):
        with pytest.raises(ValueError, match='prefix'):
            policy.forward_inf_training_time_rtc(qpos[:1], image[:1], prefix)
    untrained = small_policy(training_time_rtc=False)
    with pytest.raises(ValueError, match='trained with training-time RTC'):
        untrained.forward_inf_training_time_rtc(qpos[:1], image[:1], torch.randn(1, 2, 7))


def test_validation_uses_the_same_delayed_prefix_objective():
    policy = small_policy()
    policy.eval()
    with mock.patch.object(policy, '_sample_training_time_rtc_delays', return_value=torch.tensor([3])) as delays:
        output = policy(torch.randn(1, 7), torch.rand(1, 1, 3, 24, 32), torch.randn(1, 16, 7),
                        torch.zeros(1, 16, dtype=torch.bool))
    delays.assert_called_once()
    assert output['training_time_rtc_delay_mean'].item() == 3


def _wait_for_worker(rtc, timeout=10.0):
    deadline = time.time() + timeout
    while rtc.worker._in_flight and time.time() < deadline:
        time.sleep(0.001)


def test_direct_rtc_queries_asynchronously_while_the_pinned_prefix_executes():
    """Replays eval_bc's direct RTC loop: 6-action chunks, a 3-action prefix, 1 blended action."""
    from inference.inference_direct import (DirectRTCQueries, blend_rtc_boundary, get_action_index,
                                            get_chunk_step_offset, rtc_planned_prefix)
    torch.manual_seed(0)
    policy = small_policy()
    policy.eval()
    qpos, image = torch.randn(1, 7), torch.rand(1, 1, 3, 24, 32)
    chunk, delay, blend = 6, 3, 1
    rtc = DirectRTCQueries(policy, delay, execution_dt=1.0, first_chunk_size=chunk, query_frequency=chunk)
    plan_valid, raw, executed, start, prev_size = False, None, None, 0, chunk
    sent, submits, merges, prefixes, raws, executeds = [], [], [], {}, {}, {}
    try:
        for t in range(24):
            observe, submit, merge = rtc.schedule(t, plan_valid)
            if submit:
                planned = rtc_planned_prefix(raw, start, get_chunk_step_offset(t, chunk, chunk), delay)
                rtc.submit(qpos, image, torch.from_numpy(planned), t)
                submits.append(t)
                prefixes[t] = planned
            if (observe and not submit) or merge:
                if merge:
                    _wait_for_worker(rtc)
                    new, length = rtc.take(t)
                    merges.append(t)
                else:
                    rtc.discard()
                    new, length = policy.forward_inf(qpos, image).numpy(), 0
                plan_valid = True
                executed = (blend_rtc_boundary(executed, new, start + prev_size - length, length, blend)
                            if length else new)
                raw, start, prev_size = new, length, chunk
                raws[t], executeds[t] = raw, executed
            sent.append(executed[:, get_action_index(t, chunk, chunk, start, executed.shape[1])])
    finally:
        rtc.stop()
    # Each query goes out `delay` steps before its boundary and takes over at the boundary.
    assert submits == [3, 9, 15, 21] and merges == [6, 12, 18]
    for submitted, merged in zip(submits, merges):
        # While the policy ran, the arm executed exactly the pinned prefix (nothing skipped)...
        for i in range(delay):
            np.testing.assert_array_equal(sent[submitted + i], prefixes[submitted][:, i])
        # ...which the new chunk reproduces; it then continues from index `delay`.
        np.testing.assert_array_equal(raws[merged][:, :delay], prefixes[submitted])
        np.testing.assert_array_equal(sent[merged], executeds[merged][:, delay])  # blended
        assert not np.array_equal(executeds[merged][:, delay], raws[merged][:, delay])
        np.testing.assert_array_equal(sent[merged + 1], raws[merged][:, delay + 1])


def test_direct_rtc_late_responses_fail_closed():
    from inference.inference_direct import DirectRTCQueries
    from inference.policy_chunks import InferenceDelayExceeded
    policy = small_policy()
    policy.eval()
    qpos, image = torch.randn(1, 7), torch.rand(1, 1, 3, 24, 32)
    rtc = DirectRTCQueries(policy, 3, execution_dt=0.01, first_chunk_size=6, query_frequency=6)  # 30 ms overlap
    try:
        with mock.patch.object(policy, 'forward_inf_training_time_rtc',
                               side_effect=lambda q, i, p: time.sleep(0.1) or torch.zeros(1, 16, 7)):
            rtc.submit(qpos, image, torch.zeros(1, 3, 7), 3)
            with pytest.raises(InferenceDelayExceeded, match='not ready'):
                rtc.take(6)  # the boundary came before the response
            _wait_for_worker(rtc)
            with pytest.raises(InferenceDelayExceeded):
                rtc.take(6)  # it arrived, but after the whole overlap
    finally:
        rtc.stop()


def test_direct_rtc_warm_up_runs_on_the_worker_and_tolerates_only_a_cold_first_query():
    import threading
    from inference.inference_direct import DirectRTCQueries
    from inference.policy_chunks import InferenceDelayExceeded
    policy = small_policy()
    policy.eval()
    qpos, image = torch.randn(1, 7), torch.rand(1, 1, 3, 24, 32)
    rtc = DirectRTCQueries(policy, 3, execution_dt=0.01, first_chunk_size=6, query_frequency=6)  # 30 ms overlap
    threads, delays = [], [0.1, 0, 0]  # the worker thread's first query is cold

    def query(q, i, p):
        threads.append(threading.current_thread().name)
        time.sleep(delays[min(len(threads) - 1, 2)])
        return torch.zeros(1, 16, 7)
    try:
        with mock.patch.object(policy, 'forward_inf_training_time_rtc', side_effect=query):
            rtc.warm_up(qpos, image)
            assert threads == ['openpi-query-worker'] * 3
            assert not rtc.pending and rtc.queries == 0 and not rtc.worker.has_work_pending()
            delays[:] = [0.1, 0.1, 0.1]  # still slower than the overlap once warm: fail before motion
            threads.clear()
            with pytest.raises(InferenceDelayExceeded, match='Warm direct RTC'):
                rtc.warm_up(qpos, image)
    finally:
        rtc.stop()


def test_direct_rtc_fresh_start_discards_a_running_query():
    from inference.inference_direct import DirectRTCQueries
    policy = small_policy()
    policy.eval()
    qpos, image = torch.randn(1, 7), torch.rand(1, 1, 3, 24, 32)
    rtc = DirectRTCQueries(policy, 3, execution_dt=1.0, first_chunk_size=6, query_frequency=6)
    try:
        with mock.patch.object(policy, 'forward_inf_training_time_rtc',
                               side_effect=lambda q, i, p: time.sleep(0.1) or torch.zeros(1, 16, 7)):
            assert rtc.schedule(3, True) == (True, True, False)
            rtc.submit(qpos, image, torch.zeros(1, 3, 7), 3)
            # The arm left the plan (pause, reset or hand correction): a fresh query at the boundary,
            # only after the running query finished, and its result is dropped.
            assert rtc.schedule(6, False) == (True, False, False)
            assert not rtc.pending and not rtc.worker.has_work_pending()
            # A restarted schedule (chunk step 0) is a fresh query as well.
            rtc.submit(qpos, image, torch.zeros(1, 3, 7), 3)
            assert rtc.schedule(0, True) == (True, False, False)
            rtc.discard()
            assert not rtc.pending and not rtc.worker.has_work_pending()
        with pytest.raises(ValueError, match='prefix has 2 actions'):
            rtc.submit(qpos, image, torch.zeros(1, 2, 7), 3)
    finally:
        rtc.stop()
    with pytest.raises(ValueError, match='shorter than the executed chunks'):
        DirectRTCQueries(policy, 6, execution_dt=0.01, first_chunk_size=6, query_frequency=8)


def test_boundary_blending_crossfades_only_the_executed_actions_after_the_pinned_prefix():
    from inference.inference_direct import blend_rtc_boundary
    prev = np.arange(100, dtype=float).reshape(1, 100, 1)  # previous plan
    next_index, delay = 60, 15
    new = np.full((1, 40, 1), 500.0)
    new[:, :delay] = prev[:, next_index:next_index + delay]  # the pinned prefix
    blended = blend_rtc_boundary(prev, new, next_index, delay, 5)  # 5 actions after the handoff
    np.testing.assert_array_equal(blended[:, :delay], new[:, :delay])  # pinned: unchanged
    np.testing.assert_array_equal(blended[:, delay + 5:], new[:, delay + 5:])  # past the window
    old = prev[0, next_index + delay:next_index + delay + 5, 0]
    weights_old = (blended[0, delay:delay + 5, 0] - 500.0) / (old - 500.0)
    # Starts almost entirely on the old plan and fades into the new chunk.
    assert np.all(np.diff(weights_old) < 0) and weights_old[0] > 0.5 > weights_old[-1]
    assert np.all((blended[0, delay:delay + 5, 0] > old) & (blended[0, delay:delay + 5, 0] < 500))
    np.testing.assert_array_equal(new[:, delay:], 500.0)  # the input chunk is not modified
    # The window is limited only by the actions left after the handoff, not by the prefix length.
    blend_rtc_boundary(prev, new, next_index, delay, 25)  # previous plan up to index 60 + 15 + 25 = 100
    with pytest.raises(ValueError, match='previous plan has only 100'):
        blend_rtc_boundary(prev, new, next_index, delay, 26)
    with pytest.raises(ValueError, match='must be positive'):
        blend_rtc_boundary(prev, new, next_index, delay, 0)


def test_boundary_blending_changes_only_new_chunk_actions_after_the_handoff():
    from inference.inference_direct import blend_rtc_boundary
    prev = np.arange(100, dtype=float).reshape(1, 100, 1)
    next_index, delay, steps = 60, 10, 4
    new = np.full((1, 40, 1), 500.0)
    new[:, :delay] = prev[:, next_index:next_index + delay] + 1.0  # even a prefix differing from the old plan
    moved = np.flatnonzero(blend_rtc_boundary(prev, new, next_index, delay, steps)[0, :, 0] != new[0, :, 0])
    np.testing.assert_array_equal(moved, np.arange(delay, delay + steps))


def test_direct_start_warms_both_samplers_through_the_shared_warm_up():
    from inference.inference_direct import warm_up_policy
    from inference.policy_warmup import DEFAULT_WARMUP_STEPS
    policy = small_policy()
    policy.eval()
    qpos, image = torch.randn(1, 7), torch.rand(1, 1, 3, 24, 32)
    real_rtc = policy.forward_inf_training_time_rtc
    calls = []
    with mock.patch.object(policy, 'forward_inf', side_effect=lambda q, i: calls.append(0) or torch.zeros(1, 16, 7)), \
            mock.patch.object(policy, 'forward_inf_training_time_rtc',
                              side_effect=lambda q, i, p: calls.append(p.shape[1]) or torch.zeros(1, 16, 7)), \
            mock.patch('inference.policy_warmup.warm_up', wraps=__import__('inference.policy_warmup').policy_warmup.warm_up) as shared:
        assert warm_up_policy(policy, qpos, image, rtc_prefix_steps=4, execution_dt=0.01) is policy
    shared.assert_called_once()
    # An episode's first query has no prefix; the rest carry the trained prefix length.
    assert calls == [0] + [4] * (DEFAULT_WARMUP_STEPS - 1)
    # As in remote mode, a prefixed (asynchronous) query slower than the overlap fails at startup.
    with mock.patch.object(policy, 'forward_inf', side_effect=lambda q, i: torch.zeros(1, 16, 7)), \
            mock.patch.object(policy, 'forward_inf_training_time_rtc',
                              side_effect=lambda q, i, p: time.sleep(0.06) or torch.zeros(1, 16, 7)):
        with pytest.raises(ValueError, match='greater than the maximum inference delay'):
            warm_up_policy(policy, qpos, image, rtc_prefix_steps=4, execution_dt=0.01)  # 40 ms overlap
    calls.clear()
    calls.clear()
    with mock.patch.object(policy, 'forward_inf', side_effect=lambda q, i: calls.append(0) or torch.zeros(1, 16, 7)):
        warm_up_policy(policy, qpos, image)  # no RTC: plain queries only
    assert calls == [0] * DEFAULT_WARMUP_STEPS
    assert real_rtc(qpos, image, torch.zeros(1, 4, 7)).shape == (1, 16, 7)


def test_shared_warm_up_checks_every_query_and_rejects_negative_steps():
    from inference.policy_warmup import warm_up
    checked = []
    timings = warm_up(lambda index: {'query_total': 0.01 * (index + 1)}, 3, check=checked.append)
    assert [t['query_total'] for t in checked] == [0.01, 0.02, 0.03] and timings == checked
    assert warm_up(lambda index: pytest.fail('no queries'), 0) == []
    with pytest.raises(ValueError, match='nonnegative'):
        warm_up(lambda index: {}, -1)


def test_handoff_report_flags_a_new_plan_behind_the_old_one():
    from inference.inference_direct import chunk_handoff_metrics, format_chunk_handoff
    ramp = np.linspace(0, 1, 60)[:, None] * np.ones(7)  # the old plan moves every joint forward
    old = ramp[None]
    new = old[:, 40:].copy()  # submitted at old index 40, with a 10-action prefix
    new[:, 10:] -= 0.05  # after the handoff the new plan lags the old by 0.05 rad
    metrics = chunk_handoff_metrics(old, new, new.copy(), old[:, 49], old[:, 48], 40, 10,
                                    np.full(7, 0.03), 50, False, image_age_ms=18.0)
    assert metrics['jump'] == pytest.approx(0.05 - 1 / 59)  # one step forward, then 0.05 back
    assert metrics['new_vs_old_along'] < 0 and metrics['trailed_max'] == pytest.approx(0.03)
    line = format_chunk_handoff(metrics)
    assert line.startswith('[Direct RTC] handoff step=50') and '(behind)' in line and 'image age 18.0 ms' in line
    assert format_chunk_handoff(dict(metrics, mode='lookahead')).startswith('[Direct lookahead]')


def test_direct_run_log_keeps_console_steps_and_handoffs(tmp_path, capsys):
    import json
    import sys
    from utils.run_log import RunLog as DirectRunLog
    stdout = sys.stdout
    log = DirectRunLog('single_arm_flow', root=tmp_path)
    log.write_settings(rtc_prefix_steps=10)
    print('[Direct RTC] handoff step=50: example')
    for t in range(3):
        log.step(t=t, target=np.full(7, t, float), qpos=np.zeros(7), observed=t == 0)
    log.handoff({'step': 50, 'jump': 0.01})
    log.flush(rollout=0, demo=0)
    log.step(t=0, target=np.zeros(7), qpos=np.zeros(7), observed=True)
    log.close()
    log.close()  # idempotent (also registered with atexit)
    assert sys.stdout is stdout
    run = next(tmp_path.glob('*/*_single_arm_flow'))
    assert 'handoff step=50' in (run / 'console.log').read_text()
    steps = np.load(run / 'steps_000.npz')
    assert steps['target'].shape == (3, 7) and steps['observed'].tolist() == [True, False, False]
    assert int(steps['meta_demo']) == 0 and (run / 'steps_001.npz').is_file()
    assert json.loads((run / 'handoffs_000.json').read_text()) == [{'step': 50, 'jump': 0.01}]
    assert json.loads((run / 'run.json').read_text()) == {'rtc_prefix_steps': 10}
