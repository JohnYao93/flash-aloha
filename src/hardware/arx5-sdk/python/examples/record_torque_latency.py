"""Measure DaMiao feedforward-torque latency with raw CAN timestamps.

This is a deliberately small-signal test.  The selected joint runs with motor
Kp=0 and an explicitly selected local motor Kd while the other joints retain
their configured damping.  Smooth integer-cycle sine bursts are applied
through t_ff, and both the SDK's atomic 500 Hz telemetry and raw SocketCAN
frames are recorded.  Record one run with Kd=0 and another with Kd>0, then use
compare_torque_latency_ab.py to identify the velocity signal seen by local Kd.

Example:
    python examples/record_torque_latency.py Trossen_L5 can2 --joint 6
    python examples/record_torque_latency.py Trossen_L5 can2 --joint 6 --target-motor-kd 0.7
    python examples/record_torque_latency.py Trossen_L5 can2 --joint 6 --ab-motor-kd 0.7
    python examples/record_torque_latency.py Trossen_L5 can2 --joint 6 \
        --local-vs-host-motor-kd 0.7 --host-joint-damping 1.0
"""

from __future__ import annotations

import csv
import html
import json
import math
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
import arx5_interface as arx5
from record_cartesian_pd_motion import HOME_FINAL_SETTLE_S, _svg_plot
from tune_cartesian_pd import smoothstep


DM_POSITION_MIN = -12.5
DM_POSITION_MAX = 12.5
DM_VELOCITY_MIN = -45.0
DM_VELOCITY_MAX = 45.0
DM_TORQUE_MIN = -18.0
DM_TORQUE_MAX = 18.0
DEFAULT_FREQUENCIES_HZ = (5.0, 8.0, 12.0, 16.0, 20.0, 25.0)
# The DM MIT torque field has only 12 bits across its full configured range.
# On a J4310, 0.05 Nm is only about 13 encoded counts and leaves the high-
# frequency response too close to the returned-torque quantization.  This is
# still less than 3% of the motor's documented 7 Nm peak output torque.
DEFAULT_AMPLITUDE_NM = 0.20
DEFAULT_BURST_MIN_DURATION_S = 1.25
DEFAULT_RAMP_DURATION_S = 0.20
DEFAULT_GAP_DURATION_S = 0.20
DEFAULT_LEAD_DURATION_S = 0.50
DEFAULT_TRAVEL_LIMIT_RAD = 0.20
DEFAULT_SPEED_LIMIT_RAD_S = 1.50
DEFAULT_JOINT_LIMIT_MARGIN_RAD = 0.10
AB_STATIONARY_SPEED_RAD_S = 0.05
AB_STATIONARY_WINDOW_S = 0.30
AB_STATIONARY_TIMEOUT_S = 2.0
AB_MAX_ANCHOR_DELTA_RAD = 0.05
SO_TIMESTAMPNS = 35
SCM_TIMESTAMPNS = SO_TIMESTAMPNS
CAN_FRAME = struct.Struct("=IB3x8s")
CAN_EFF_MASK = 0x1FFFFFFF


def uint_to_float(value: int, minimum: float, maximum: float, bits: int) -> float:
    return float(value) * (maximum - minimum) / ((1 << bits) - 1) + minimum


def decode_dm_command(data: bytes) -> dict[str, float | int]:
    """Decode the standard 8-byte DaMiao MIT command payload."""
    if len(data) != 8:
        raise ValueError("DaMiao command must contain eight bytes")
    position_raw = (data[0] << 8) | data[1]
    velocity_raw = (data[2] << 4) | (data[3] >> 4)
    kp_raw = ((data[3] & 0x0F) << 8) | data[4]
    kd_raw = (data[5] << 4) | (data[6] >> 4)
    torque_raw = ((data[6] & 0x0F) << 8) | data[7]
    return {
        "position_raw": position_raw,
        "velocity_raw": velocity_raw,
        "kp_raw": kp_raw,
        "kd_raw": kd_raw,
        "torque_raw": torque_raw,
        "position": uint_to_float(position_raw, DM_POSITION_MIN, DM_POSITION_MAX, 16),
        "velocity": uint_to_float(velocity_raw, DM_VELOCITY_MIN, DM_VELOCITY_MAX, 12),
        "kp": uint_to_float(kp_raw, 0.0, 500.0, 12),
        "kd": uint_to_float(kd_raw, 0.0, 5.0, 12),
        "torque": uint_to_float(torque_raw, DM_TORQUE_MIN, DM_TORQUE_MAX, 12),
    }


def decode_dm_feedback(data: bytes) -> dict[str, float | int]:
    """Decode the standard 8-byte DaMiao position/velocity/torque reply."""
    if len(data) != 8:
        raise ValueError("DaMiao feedback must contain eight bytes")
    motor_id = data[0] & 0x0F
    error = data[0] >> 4
    position_raw = (data[1] << 8) | data[2]
    velocity_raw = (data[3] << 4) | (data[4] >> 4)
    torque_raw = ((data[4] & 0x0F) << 8) | data[5]
    return {
        "motor_id": motor_id,
        "error": error,
        "position_raw": position_raw,
        "velocity_raw": velocity_raw,
        "torque_raw": torque_raw,
        "position": uint_to_float(position_raw, DM_POSITION_MIN, DM_POSITION_MAX, 16),
        "velocity": uint_to_float(velocity_raw, DM_VELOCITY_MIN, DM_VELOCITY_MAX, 12),
        "torque": uint_to_float(torque_raw, DM_TORQUE_MIN, DM_TORQUE_MAX, 12),
        "mos_temperature_c": data[6],
        "rotor_temperature_c": data[7],
    }


@dataclass(frozen=True)
class Burst:
    frequency_hz: float
    start_s: float
    duration_s: float


class TorqueBurstProfile:
    """Bounded integer-cycle sine bursts with smooth amplitude envelopes."""

    def __init__(
        self,
        amplitude_nm=DEFAULT_AMPLITUDE_NM,
        frequencies_hz=DEFAULT_FREQUENCIES_HZ,
        minimum_duration_s=DEFAULT_BURST_MIN_DURATION_S,
        ramp_duration_s=DEFAULT_RAMP_DURATION_S,
        gap_duration_s=DEFAULT_GAP_DURATION_S,
        lead_duration_s=DEFAULT_LEAD_DURATION_S,
    ):
        self.amplitude_nm = float(amplitude_nm)
        self.ramp_duration_s = float(ramp_duration_s)
        self.gap_duration_s = float(gap_duration_s)
        self.lead_duration_s = float(lead_duration_s)
        if not np.isfinite(self.amplitude_nm) or self.amplitude_nm <= 0.0:
            raise ValueError("amplitude must be finite and positive")
        if min(minimum_duration_s, ramp_duration_s, gap_duration_s, lead_duration_s) < 0.0:
            raise ValueError("profile durations must be nonnegative")
        self.bursts: list[Burst] = []
        start = self.lead_duration_s
        for frequency in frequencies_hz:
            frequency = float(frequency)
            if not np.isfinite(frequency) or frequency <= 0.0:
                raise ValueError("frequencies must be finite and positive")
            cycles = max(1, math.ceil(float(minimum_duration_s) * frequency))
            duration = cycles / frequency
            if 2.0 * self.ramp_duration_s >= duration:
                raise ValueError("each burst must be longer than twice its ramp")
            self.bursts.append(Burst(frequency, start, duration))
            start += duration + self.gap_duration_s
        self.duration_s = start + self.lead_duration_s

    def sample(self, elapsed_s: float) -> tuple[float, float, float]:
        elapsed_s = float(elapsed_s)
        for burst in self.bursts:
            local_time = elapsed_s - burst.start_s
            if 0.0 <= local_time < burst.duration_s:
                ramp_in = smoothstep(local_time / self.ramp_duration_s)
                ramp_out = smoothstep(
                    (burst.duration_s - local_time) / self.ramp_duration_s
                )
                envelope = min(ramp_in, ramp_out)
                torque = (
                    self.amplitude_nm
                    * envelope
                    * math.sin(2.0 * math.pi * burst.frequency_hz * local_time)
                )
                return torque, burst.frequency_hz, envelope
        return 0.0, 0.0, 0.0


@dataclass
class RawCanFrame:
    kernel_realtime_ns: int
    kernel_monotonic_ns: int
    host_receive_monotonic_ns: int
    can_id: int
    dlc: int
    data: bytes
    message_flags: int
    locally_generated: bool


class RawCanRecorder:
    """Capture SocketCAN frames with kernel software receive timestamps."""

    def __init__(self, interface: str):
        self.interface = interface
        self.frames: list[RawCanFrame] = []
        self.error: Exception | None = None
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clock_offset_ns = 0

    def start(self) -> None:
        before_mono = time.monotonic_ns()
        realtime = time.time_ns()
        after_mono = time.monotonic_ns()
        self._clock_offset_ns = (before_mono + after_mono) // 2 - realtime
        can_socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        can_socket.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        can_socket.settimeout(0.05)
        can_socket.bind((self.interface,))
        self._socket = can_socket
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._socket is not None
        try:
            while not self._stop.is_set():
                try:
                    packet, ancillary, flags, _ = self._socket.recvmsg(
                        CAN_FRAME.size, 128
                    )
                except socket.timeout:
                    continue
                if len(packet) != CAN_FRAME.size:
                    continue
                receive_mono = time.monotonic_ns()
                can_id, dlc, payload = CAN_FRAME.unpack(packet)
                kernel_realtime_ns = 0
                for level, kind, value in ancillary:
                    if (
                        level == socket.SOL_SOCKET
                        and kind == SCM_TIMESTAMPNS
                        and len(value) >= struct.calcsize("@qq")
                    ):
                        seconds, nanoseconds = struct.unpack_from("@qq", value)
                        kernel_realtime_ns = seconds * 1_000_000_000 + nanoseconds
                        break
                kernel_mono = (
                    kernel_realtime_ns + self._clock_offset_ns
                    if kernel_realtime_ns
                    else receive_mono
                )
                self.frames.append(
                    RawCanFrame(
                        kernel_realtime_ns,
                        kernel_mono,
                        receive_mono,
                        can_id & CAN_EFF_MASK,
                        min(int(dlc), 8),
                        payload[: min(int(dlc), 8)],
                        flags,
                        bool(flags & socket.MSG_DONTROUTE),
                    )
                )
        except Exception as exc:  # surfaced after the robot is made safe
            self.error = exc

    def stop(self) -> list[RawCanFrame]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._socket is not None:
            self._socket.close()
        if self.error is not None:
            raise RuntimeError(f"raw CAN capture failed: {self.error}")
        return self.frames


def motor_torque_constant(motor_type) -> float:
    if motor_type == arx5.MotorType.DM_J4310:
        return 0.424
    if motor_type == arx5.MotorType.DM_J4340:
        return 1.0
    raise ValueError("raw latency test currently supports DaMiao motors only")


def safety_guard_reason(
    position,
    velocity,
    anchor_position,
    joint_min,
    joint_max,
    target_index,
    travel_limit_rad,
    speed_limit_rad_s,
    joint_limit_margin_rad,
) -> str | None:
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        return "non-finite joint feedback"
    travel = abs(position[target_index] - anchor_position[target_index])
    if travel > travel_limit_rad:
        return f"joint {target_index + 1} moved {travel:.3f} rad from its anchor"
    if abs(velocity[target_index]) > speed_limit_rad_s:
        return (
            f"joint {target_index + 1} reached "
            f"{abs(velocity[target_index]):.3f} rad/s"
        )
    clearance = np.minimum(position - joint_min, joint_max - position)
    if np.min(clearance) <= joint_limit_margin_rad:
        index = int(np.argmin(clearance))
        return f"joint {index + 1} approached its position limit"
    return None


def wait_for_stationary_arm(
    controller,
    speed_threshold_rad_s=AB_STATIONARY_SPEED_RAD_S,
    stationary_window_s=AB_STATIONARY_WINDOW_S,
    timeout_s=AB_STATIONARY_TIMEOUT_S,
):
    """Wait until every arm joint remains slow, without issuing a new command."""
    deadline = time.monotonic() + timeout_s
    stationary_since = None
    maximum_speed = 0.0
    while time.monotonic() < deadline:
        velocity = np.asarray(controller.get_joint_state().vel(), dtype=np.float64)
        speed = float(np.max(np.abs(velocity)))
        maximum_speed = max(maximum_speed, speed)
        now = time.monotonic()
        if speed <= speed_threshold_rad_s:
            if stationary_since is None:
                stationary_since = now
            elif now - stationary_since >= stationary_window_s:
                return maximum_speed
        else:
            stationary_since = None
        time.sleep(0.01)
    raise RuntimeError(
        f"arm did not settle below {speed_threshold_rad_s:g} rad/s"
    )


def frames_in_monotonic_window(frames, start_ns, end_ns):
    return [
        frame
        for frame in frames
        if start_ns <= frame.kernel_monotonic_ns <= end_ns
    ]


def motor_kd_condition_specs(
    target_motor_kd,
    ab_motor_kd,
    local_vs_host_motor_kd=None,
    host_joint_damping=1.0,
):
    """Return execution order; damped-first avoids recovering from Kd=0 drift."""
    if ab_motor_kd is not None and local_vs_host_motor_kd is not None:
        raise ValueError("select only one same-session comparison mode")
    if local_vs_host_motor_kd is not None:
        if abs(float(target_motor_kd)) > 1e-12:
            raise ValueError(
                "target motor Kd must remain zero in local-vs-host mode"
            )
        return [
            ("local", float(local_vs_host_motor_kd), 0.0),
            ("host", 0.0, float(host_joint_damping)),
        ]
    if ab_motor_kd is None:
        return [("single", float(target_motor_kd), 0.0)]
    if abs(float(target_motor_kd)) > 1e-12:
        raise ValueError(
            "target motor Kd must remain zero in same-session A/B mode"
        )
    return [("damped", float(ab_motor_kd), 0.0), ("baseline", 0.0, 0.0)]


def telemetry_columns(joint_dof=6) -> list[str]:
    columns = [
        "host_elapsed_s",
        "host_monotonic_ns",
        "profile_torque_nm",
        "burst_frequency_hz",
        "burst_envelope",
        "command_call_duration_s",
        "control_cycle_sequence",
        "control_cycle_start_s",
        "feedback_used_timestamp_s",
        "feedback_publish_timestamp_s",
        "control_cycle_end_s",
    ]
    for prefix in (
        "control_joint_velocity",
        "host_joint_damping_torque",
        "joint_position",
        "joint_velocity",
        "joint_torque_measured",
        "joint_position_command",
        "joint_velocity_command",
        "joint_torque_command",
        "joint_send_call_start_s",
        "joint_send_call_end_s",
    ):
        columns.extend(f"{prefix}_{index + 1}" for index in range(joint_dof))
    return columns


def capture_telemetry_row(
    controller,
    host_elapsed_s,
    host_monotonic_ns,
    profile_torque_nm,
    burst_frequency_hz,
    burst_envelope,
    command_call_duration_s,
    host_joint_damping=None,
) -> list[float]:
    telemetry = controller._get_control_cycle_telemetry()
    row = [
        host_elapsed_s,
        host_monotonic_ns,
        profile_torque_nm,
        burst_frequency_hz,
        burst_envelope,
        command_call_duration_s,
        telemetry.sequence,
        telemetry.cycle_start_s,
        telemetry.feedback_used_timestamp_s,
        telemetry.feedback_publish_timestamp_s,
        telemetry.cycle_end_s,
    ]
    host_joint_damping = (
        np.zeros(6, dtype=np.float64)
        if host_joint_damping is None
        else np.asarray(host_joint_damping, dtype=np.float64)
    )
    host_damping_torque = (
        -host_joint_damping
        * np.asarray(telemetry.control_joint_velocity, dtype=np.float64)
    )
    for values in (
        telemetry.control_joint_velocity,
        host_damping_torque,
        telemetry.feedback_joint_position,
        telemetry.feedback_joint_velocity,
        telemetry.feedback_joint_torque,
        telemetry.final_joint_position_command,
        telemetry.final_joint_velocity_command,
        telemetry.final_joint_torque_command,
        telemetry.joint_send_call_start_s,
        telemetry.joint_send_call_end_s,
    ):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise RuntimeError("controller telemetry contains an invalid joint vector")
        row.extend(array.tolist())
    return [float(value) for value in row]


def classify_target_frames(
    frames: list[RawCanFrame],
    target_motor_id: int,
    torque_constant: float,
    velocity_feedback_scale: float = 1.0,
) -> tuple[list[dict], list[dict]]:
    commands = []
    feedback = []
    for frame in frames:
        if frame.dlc != 8:
            continue
        if frame.locally_generated and frame.can_id == target_motor_id:
            decoded = decode_dm_command(frame.data)
            commands.append(
                {
                    "time_ns": frame.kernel_monotonic_ns,
                    "torque_raw": decoded["torque_raw"],
                    "torque_motor_units": decoded["torque"],
                    "torque_sdk_nm": decoded["torque"] * torque_constant,
                    "kp": decoded["kp"],
                    "kd": decoded["kd"],
                }
            )
        elif not frame.locally_generated:
            decoded = decode_dm_feedback(frame.data)
            if decoded["motor_id"] == target_motor_id:
                feedback.append(
                    {
                        "time_ns": frame.kernel_monotonic_ns,
                        "torque_raw": decoded["torque_raw"],
                        "torque_motor_units": decoded["torque"],
                        "torque_sdk_nm": decoded["torque"] * torque_constant,
                        "position": decoded["position"],
                        "velocity": decoded["velocity"],
                        "velocity_sdk_rad_s": (
                            decoded["velocity"] * velocity_feedback_scale
                        ),
                        "error": decoded["error"],
                    }
                )
    return commands, feedback


def _phasor(values, times, frequency_hz):
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    window = np.hanning(values.size)
    centered = values - np.sum(window * values) / np.sum(window)
    return np.sum(
        window * centered * np.exp(-2j * np.pi * frequency_hz * times)
    )


def _step_resample(times, values, query):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    indices = np.searchsorted(times, query, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 1)
    return values[indices]


def fit_first_order_lag_plus_delay(frequencies_hz, response):
    """Fit H(jw) = K / (1 + jw*tau) * exp(-jw*delay).

    The magnitude identifies the first-order time constant without confusing
    its phase lag with transport delay.  Delay is then fit to the residual
    unwrapped phase with the physically meaningful zero-phase DC constraint.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    response = np.asarray(response, dtype=np.complex128)
    valid = (
        np.isfinite(frequencies)
        & np.isfinite(response.real)
        & np.isfinite(response.imag)
        & (frequencies > 0.0)
    )
    frequencies = frequencies[valid]
    response = response[valid]
    if frequencies.size < 3:
        return None

    omega = 2.0 * np.pi * frequencies
    magnitude = np.abs(response)
    # A dense one-dimensional search is deterministic, dependency-free, and
    # inexpensive for the handful of burst frequencies in this experiment.
    time_constants = np.linspace(0.0, 0.050, 10001)
    bases = 1.0 / np.sqrt(
        1.0 + np.square(omega[:, np.newaxis] * time_constants[np.newaxis, :])
    )
    gains = (bases.T @ magnitude) / np.sum(np.square(bases), axis=0)
    magnitude_errors = magnitude[:, np.newaxis] - bases * gains
    best_index = int(np.argmin(np.mean(np.square(magnitude_errors), axis=0)))
    time_constant = float(time_constants[best_index])
    dc_gain = float(gains[best_index])

    measured_phase = np.unwrap(np.angle(response))
    lag_phase = -np.arctan(omega * time_constant)
    delay = float(
        -np.dot(omega, measured_phase - lag_phase) / np.dot(omega, omega)
    )
    delay = max(0.0, delay)
    predicted = (
        dc_gain
        / (1.0 + 1j * omega * time_constant)
        * np.exp(-1j * omega * delay)
    )
    phase_residual = np.angle(response / predicted)
    cutoff_hz = (
        float(1.0 / (2.0 * np.pi * time_constant))
        if time_constant > 0.0
        else None
    )
    return {
        "dc_gain": dc_gain,
        "time_constant_ms": time_constant * 1000.0,
        "cutoff_frequency_hz": cutoff_hz,
        "additional_delay_ms": delay * 1000.0,
        "magnitude_rmse": float(
            np.sqrt(np.mean(np.square(np.abs(response) - np.abs(predicted))))
        ),
        "phase_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(np.square(phase_residual))))
        ),
    }


def _complex_from_gain_phase(gain, phase_deg):
    return float(gain) * np.exp(1j * np.deg2rad(float(phase_deg)))


def _wrapped_phase_difference_deg(first, second):
    return float(np.angle(np.exp(1j * (np.angle(first) - np.angle(second))), deg=True))


def compare_motor_kd_frequency_response(baseline, damped):
    """Infer the local-Kd velocity path from two torque-burst metadata dicts.

    If G0 = V_reported / T_ff with local Kd off and Gd is the same transfer
    with damping B enabled, then

        H_internal / H_reported = (G0 / Gd - 1) / (B * G0).

    A ratio near 1 means local Kd sees substantially the same filtered velocity
    as the host.  A ratio near 1/H_reported, where H_reported is measured against
    j*w*position, means local Kd sees a fresher, less-filtered velocity estimate.
    """
    for field in ("model", "joint", "motor_id"):
        if baseline.get(field) != damped.get(field):
            raise ValueError(
                f"recordings have different {field}: "
                f"{baseline.get(field)!r} vs {damped.get(field)!r}"
            )
    for label, recording in (("baseline", baseline), ("damped", damped)):
        if recording.get("failure"):
            raise ValueError(f"{label} recording ended with: {recording['failure']}")
        status_codes = recording.get("analysis", {}).get(
            "raw_feedback_status_codes",
            recording.get("analysis", {}).get("raw_feedback_error_codes", []),
        )
        invalid_status_codes = [
            int(code) for code in status_codes if int(code) not in (0, 1)
        ]
        if invalid_status_codes:
            raise ValueError(
                f"{label} recording contains abnormal motor status codes "
                f"{invalid_status_codes}"
            )
        raw_kp = recording.get("analysis", {}).get("raw_command_kp_median")
        if raw_kp is not None and abs(float(raw_kp)) > 0.15:
            raise ValueError(f"{label} recording did not use motor Kp=0")
    baseline_anchor = baseline.get("anchor_joint_position_rad")
    damped_anchor = damped.get("anchor_joint_position_rad")
    maximum_anchor_difference = None
    if baseline_anchor is not None and damped_anchor is not None:
        baseline_anchor = np.asarray(baseline_anchor, dtype=np.float64)
        damped_anchor = np.asarray(damped_anchor, dtype=np.float64)
        if baseline_anchor.shape != damped_anchor.shape:
            raise ValueError("recordings have incompatible anchor joint vectors")
        maximum_anchor_difference = float(
            np.max(np.abs(baseline_anchor - damped_anchor))
        )
        if maximum_anchor_difference > 0.05:
            raise ValueError(
                "recording anchors differ by "
                f"{maximum_anchor_difference:.3f} rad; repeat at the same pose"
            )
    baseline_amplitude = float(baseline["amplitude_nm"])
    damped_amplitude = float(damped["amplitude_nm"])
    if not math.isclose(
        baseline_amplitude, damped_amplitude, rel_tol=0.02, abs_tol=1e-9
    ):
        raise ValueError("recordings must use the same torque amplitude (within 2%)")

    baseline_analysis = baseline.get("analysis", {})
    damped_analysis = damped.get("analysis", {})
    baseline_kd = baseline_analysis.get(
        "raw_command_kd_median", baseline.get("selected_joint_motor_kd")
    )
    damped_kd = damped_analysis.get(
        "raw_command_kd_median", damped.get("selected_joint_motor_kd")
    )
    if baseline_kd is None or damped_kd is None:
        raise ValueError("recordings do not contain selected motor Kd")
    baseline_kd = float(baseline_kd)
    damped_kd = float(damped_kd)
    if abs(baseline_kd) > 0.005:
        raise ValueError("baseline recording must have local motor Kd=0")
    if damped_kd <= baseline_kd + 0.005:
        raise ValueError("damped recording must use a meaningfully larger local motor Kd")

    baseline_torque_constant = float(baseline["torque_constant"])
    damped_torque_constant = float(damped["torque_constant"])
    if not math.isclose(
        baseline_torque_constant, damped_torque_constant, rel_tol=1e-6
    ):
        raise ValueError("recordings have different motor torque constants")
    effective_damping = baseline_torque_constant * (damped_kd - baseline_kd)

    def keyed_response(metadata):
        result = {}
        for item in metadata.get("analysis", {}).get("frequency_response", []):
            required = (
                "raw_velocity_gain_rad_s_per_nm",
                "raw_velocity_phase_deg",
                "reported_velocity_vs_position_derivative_gain",
                "reported_velocity_vs_position_derivative_phase_deg",
            )
            if all(name in item for name in required):
                result[round(float(item["frequency_hz"]), 9)] = item
        return result

    baseline_response = keyed_response(baseline)
    damped_response = keyed_response(damped)
    common_frequencies = sorted(set(baseline_response) & set(damped_response))
    if len(common_frequencies) < 3:
        raise ValueError(
            "recordings need at least three matching frequency responses with "
            "raw velocity fields; record both runs with the updated recorder"
        )

    rows = []
    same_phase_errors = []
    fresh_phase_errors = []
    damping_effects = []
    for frequency in common_frequencies:
        base_item = baseline_response[frequency]
        damped_item = damped_response[frequency]
        g0 = _complex_from_gain_phase(
            base_item["raw_velocity_gain_rad_s_per_nm"],
            base_item["raw_velocity_phase_deg"],
        )
        gd = _complex_from_gain_phase(
            damped_item["raw_velocity_gain_rad_s_per_nm"],
            damped_item["raw_velocity_phase_deg"],
        )
        reported_filter = _complex_from_gain_phase(
            base_item["reported_velocity_vs_position_derivative_gain"],
            base_item["reported_velocity_vs_position_derivative_phase_deg"],
        )
        if min(abs(g0), abs(gd), abs(reported_filter)) < 1e-12:
            continue
        damping_effect = float(abs(g0 / gd - 1.0))
        inferred = (g0 / gd - 1.0) / (effective_damping * g0)
        expected_same = 1.0 + 0.0j
        expected_fresh = 1.0 / reported_filter
        same_error = _wrapped_phase_difference_deg(inferred, expected_same)
        fresh_error = _wrapped_phase_difference_deg(inferred, expected_fresh)
        same_phase_errors.append(same_error)
        fresh_phase_errors.append(fresh_error)
        damping_effects.append(damping_effect)
        rows.append(
            {
                "frequency_hz": frequency,
                "baseline_velocity_gain_rad_s_per_nm": float(abs(g0)),
                "baseline_velocity_phase_deg": float(np.angle(g0, deg=True)),
                "damped_velocity_gain_rad_s_per_nm": float(abs(gd)),
                "damped_velocity_phase_deg": float(np.angle(gd, deg=True)),
                "damping_effect_fraction": damping_effect,
                "inferred_internal_vs_reported_gain": float(abs(inferred)),
                "inferred_internal_vs_reported_phase_deg": float(
                    np.angle(inferred, deg=True)
                ),
                "same_reported_velocity_expected_gain": 1.0,
                "same_reported_velocity_expected_phase_deg": 0.0,
                "fresher_velocity_expected_gain": float(abs(expected_fresh)),
                "fresher_velocity_expected_phase_deg": float(
                    np.angle(expected_fresh, deg=True)
                ),
                "same_reported_velocity_phase_error_deg": same_error,
                "fresher_velocity_phase_error_deg": fresh_error,
            }
        )
    if len(rows) < 3:
        raise ValueError("too few finite matching velocity responses")

    same_rmse = float(np.sqrt(np.mean(np.square(same_phase_errors))))
    fresh_rmse = float(np.sqrt(np.mean(np.square(fresh_phase_errors))))
    margin = abs(same_rmse - fresh_rmse)
    median_damping_effect = float(np.median(damping_effects))
    if median_damping_effect < 0.05:
        closer = "inconclusive_weak_damping_effect"
    elif margin < 5.0:
        closer = "inconclusive"
    elif same_rmse < fresh_rmse:
        closer = "same_reported_velocity"
    else:
        closer = "fresher_less_filtered_velocity"
    return {
        "model": baseline["model"],
        "joint": baseline["joint"],
        "motor_id": baseline["motor_id"],
        "baseline_motor_kd": baseline_kd,
        "damped_motor_kd": damped_kd,
        "assumed_effective_damping_increment_nm_s_per_rad": effective_damping,
        "maximum_anchor_difference_rad": maximum_anchor_difference,
        "equation": "H_internal/H_reported = (G0/Gd - 1)/(B*G0)",
        "frequency_response": rows,
        "summary": {
            "same_reported_velocity_phase_rmse_deg": same_rmse,
            "fresher_velocity_phase_rmse_deg": fresh_rmse,
            "phase_rmse_margin_deg": margin,
            "median_damping_effect_fraction": median_damping_effect,
            "closer_phase_hypothesis": closer,
            "interpretation_note": (
                "The hypothesis decision uses phase, which is invariant to an "
                "unknown positive scale in DaMiao Kd. Unmodeled friction, "
                "nonlinearity, pose drift, and low signal-to-noise can still "
                "make the result inconclusive."
            ),
        },
    }


def compare_local_vs_host_damping(local, host):
    """Compare matched local and 500 Hz host joint damping conditions."""
    for field in ("model", "joint", "motor_id"):
        if local.get(field) != host.get(field):
            raise ValueError(f"local and host recordings have different {field}")
    for label, recording in (("local", local), ("host", host)):
        if recording.get("failure"):
            raise ValueError(f"{label} recording ended with: {recording['failure']}")
        status_codes = recording.get("analysis", {}).get(
            "raw_feedback_status_codes",
            recording.get("analysis", {}).get("raw_feedback_error_codes", []),
        )
        if any(int(code) not in (0, 1) for code in status_codes):
            raise ValueError(f"{label} recording contains an abnormal motor status")
    local_kd = float(local["analysis"]["raw_command_kd_median"])
    host_kd = float(host["analysis"]["raw_command_kd_median"])
    host_damping = float(host.get("host_joint_damping_nm_s_per_rad", 0.0))
    if local_kd <= 0.005 or abs(host_kd) > 0.005 or host_damping <= 0.0:
        raise ValueError(
            "expected local motor Kd>0 and host motor Kd=0 with host damping>0"
        )
    local_anchor = np.asarray(local["anchor_joint_position_rad"], dtype=np.float64)
    host_anchor = np.asarray(host["anchor_joint_position_rad"], dtype=np.float64)
    anchor_delta = float(np.max(np.abs(local_anchor - host_anchor)))
    if anchor_delta > AB_MAX_ANCHOR_DELTA_RAD:
        raise ValueError(
            f"local/host anchors differ by {anchor_delta:.3f} rad"
        )

    def keyed(recording):
        return {
            round(float(item["frequency_hz"]), 9): item
            for item in recording["analysis"]["frequency_response"]
            if "excitation_velocity_gain_rad_s_per_nm" in item
        }

    local_response = keyed(local)
    host_response = keyed(host)
    frequencies = sorted(set(local_response) & set(host_response))
    if len(frequencies) < 3:
        raise ValueError("local and host recordings need three matching responses")
    rows = []
    phase_differences = []
    log_gain_ratios = []
    for frequency in frequencies:
        local_item = local_response[frequency]
        host_item = host_response[frequency]
        local_transfer = _complex_from_gain_phase(
            local_item["excitation_velocity_gain_rad_s_per_nm"],
            local_item["excitation_velocity_phase_deg"],
        )
        host_transfer = _complex_from_gain_phase(
            host_item["excitation_velocity_gain_rad_s_per_nm"],
            host_item["excitation_velocity_phase_deg"],
        )
        if min(abs(local_transfer), abs(host_transfer)) < 1e-12:
            continue
        ratio = host_transfer / local_transfer
        phase_difference = float(np.angle(ratio, deg=True))
        gain_ratio = float(abs(ratio))
        phase_differences.append(phase_difference)
        log_gain_ratios.append(math.log(gain_ratio))
        rows.append(
            {
                "frequency_hz": frequency,
                "local_velocity_gain_rad_s_per_nm": float(abs(local_transfer)),
                "local_velocity_phase_deg": float(
                    np.angle(local_transfer, deg=True)
                ),
                "host_velocity_gain_rad_s_per_nm": float(abs(host_transfer)),
                "host_velocity_phase_deg": float(
                    np.angle(host_transfer, deg=True)
                ),
                "host_over_local_gain": gain_ratio,
                "host_minus_local_phase_deg": phase_difference,
                "host_minus_local_equivalent_delay_ms": (
                    -phase_difference / (360.0 * frequency) * 1000.0
                ),
            }
        )
    if len(rows) < 3:
        raise ValueError("too few finite local/host responses")
    return {
        "model": local["model"],
        "joint": local["joint"],
        "motor_id": local["motor_id"],
        "local_motor_kd": local_kd,
        "host_joint_damping_nm_s_per_rad": host_damping,
        "maximum_anchor_difference_rad": anchor_delta,
        "frequency_response": rows,
        "summary": {
            "median_host_over_local_gain": float(
                np.exp(np.median(log_gain_ratios))
            ),
            "host_minus_local_phase_rms_deg": float(
                np.sqrt(np.mean(np.square(phase_differences)))
            ),
            "median_host_minus_local_phase_deg": float(
                np.median(phase_differences)
            ),
            "interpretation_note": (
                "Gain above one means host damping permits more velocity for "
                "the same excitation. Negative phase means the host response "
                "lags the local-damping response."
            ),
        },
    }


def analyze_latency(
    profile: TorqueBurstProfile,
    telemetry_rows,
    frames,
    target_index,
    target_motor_id,
    torque_constant,
    velocity_feedback_scale,
    experiment_start_ns,
):
    commands, feedback = classify_target_frames(
        frames, target_motor_id, torque_constant, velocity_feedback_scale
    )
    torque_lsb_nm = float(
        torque_constant * (DM_TORQUE_MAX - DM_TORQUE_MIN) / ((1 << 12) - 1)
    )
    result = {
        "raw_command_frame_count": len(commands),
        "raw_feedback_frame_count": len(feedback),
        "raw_command_kp_median": (
            float(np.median([item["kp"] for item in commands]))
            if commands
            else None
        ),
        "raw_command_kd_median": (
            float(np.median([item["kd"] for item in commands]))
            if commands
            else None
        ),
        "raw_feedback_status_codes": sorted(
            {int(item["error"]) for item in feedback}
        ),
        # Kept for compatibility with recordings made before the field was
        # correctly identified as a status nibble.  Values 0 (disabled) and 1
        # (enabled) are normal; DaMiao fault states are 8 through 14.
        "raw_feedback_error_codes": sorted(
            {int(item["error"]) for item in feedback}
        ),
        "packet_turnaround_ms": None,
        "torque_quantization_lsb_nm": torque_lsb_nm,
        "command_peak_quantization_counts": float(
            profile.amplitude_nm / torque_lsb_nm
        ),
        "frequency_response": [],
        "phase_fit": None,
        "first_order_lag_plus_delay_fit": None,
        "reported_velocity_filter_fit": None,
    }
    if not commands or not feedback or not telemetry_rows:
        return result

    command_times = np.array([item["time_ns"] for item in commands]) * 1e-9
    feedback_times = np.array([item["time_ns"] for item in feedback]) * 1e-9
    command_values = np.array([item["torque_sdk_nm"] for item in commands])
    feedback_values = np.array([item["torque_sdk_nm"] for item in feedback])
    feedback_velocity_values = np.array(
        [item["velocity_sdk_rad_s"] for item in feedback]
    )
    feedback_position_values = np.array(
        [item["position"] for item in feedback]
    )
    next_feedback = np.searchsorted(feedback_times, command_times, side="left")
    valid = next_feedback < feedback_times.size
    turnaround = 1000.0 * (
        feedback_times[next_feedback[valid]] - command_times[valid]
    )
    nonnegative = turnaround[turnaround >= 0.0]
    if nonnegative.size:
        result["packet_turnaround_ms"] = {
            "median": float(np.median(nonnegative)),
            "p95": float(np.percentile(nonnegative, 95)),
            "maximum": float(np.max(nonnegative)),
        }

    columns = telemetry_columns()
    data = np.asarray(telemetry_rows, dtype=np.float64)
    ci = {name: index for index, name in enumerate(columns)}
    telemetry_time = data[:, ci["host_monotonic_ns"]] * 1e-9
    joint_position = data[:, ci[f"joint_position_{target_index + 1}"]]
    final_command = data[:, ci[f"joint_torque_command_{target_index + 1}"]]
    measured_torque = data[:, ci[f"joint_torque_measured_{target_index + 1}"]]
    profile_torque = data[:, ci["profile_torque_nm"]]
    start_s = experiment_start_ns * 1e-9

    phases = []
    frequencies = []
    raw_responses = []
    reported_velocity_filter_responses = []
    for burst in profile.bursts:
        plateau_start = start_s + burst.start_s + profile.ramp_duration_s
        plateau_end = start_s + burst.start_s + burst.duration_s - profile.ramp_duration_s
        query = np.arange(plateau_start, plateau_end, 0.002)
        if query.size < 32:
            continue
        raw_command = _step_resample(command_times, command_values, query)
        raw_feedback = _step_resample(feedback_times, feedback_values, query)
        raw_velocity = _step_resample(
            feedback_times, feedback_velocity_values, query
        )
        raw_position = _step_resample(
            feedback_times, feedback_position_values, query
        )
        telem_command = np.interp(query, telemetry_time, final_command)
        telem_feedback = np.interp(query, telemetry_time, measured_torque)
        excitation = np.interp(query, telemetry_time, profile_torque)
        position = np.interp(query, telemetry_time, joint_position)
        relative_time = query - query[0]
        command_phasor = _phasor(raw_command, relative_time, burst.frequency_hz)
        feedback_phasor = _phasor(raw_feedback, relative_time, burst.frequency_hz)
        velocity_phasor = _phasor(
            raw_velocity, relative_time, burst.frequency_hz
        )
        raw_position_phasor = _phasor(
            raw_position, relative_time, burst.frequency_hz
        )
        telem_command_phasor = _phasor(
            telem_command, relative_time, burst.frequency_hz
        )
        telem_feedback_phasor = _phasor(
            telem_feedback, relative_time, burst.frequency_hz
        )
        excitation_phasor = _phasor(
            excitation, relative_time, burst.frequency_hz
        )
        position_phasor = _phasor(position, relative_time, burst.frequency_hz)
        if abs(command_phasor) < 1e-12:
            continue
        raw_ratio = feedback_phasor / command_phasor
        telemetry_ratio = (
            telem_feedback_phasor / telem_command_phasor
            if abs(telem_command_phasor) >= 1e-12
            else complex(np.nan, np.nan)
        )
        acceleration_phasor = -(
            2.0 * np.pi * burst.frequency_hz
        ) ** 2 * position_phasor
        acceleration_ratio = acceleration_phasor / command_phasor
        velocity_ratio = velocity_phasor / command_phasor
        excitation_velocity_ratio = (
            velocity_phasor / excitation_phasor
            if abs(excitation_phasor) >= 1e-12
            else complex(np.nan, np.nan)
        )
        position_derivative_phasor = (
            1j * 2.0 * np.pi * burst.frequency_hz * raw_position_phasor
        )
        velocity_vs_position_derivative = (
            velocity_phasor / position_derivative_phasor
            if abs(position_derivative_phasor) >= 1e-12
            else complex(np.nan, np.nan)
        )
        raw_phase_deg = float(np.angle(raw_ratio, deg=True))
        frequencies.append(burst.frequency_hz)
        phases.append(raw_phase_deg)
        raw_responses.append(raw_ratio)
        reported_velocity_filter_responses.append(
            velocity_vs_position_derivative
        )
        result["frequency_response"].append(
            {
                "frequency_hz": burst.frequency_hz,
                "raw_torque_gain": float(abs(raw_ratio)),
                "raw_feedback_amplitude_nm": float(
                    profile.amplitude_nm * abs(raw_ratio)
                ),
                "raw_feedback_amplitude_counts": float(
                    profile.amplitude_nm
                    * abs(raw_ratio)
                    / result["torque_quantization_lsb_nm"]
                ),
                "raw_torque_phase_deg": raw_phase_deg,
                "raw_torque_equivalent_delay_ms": (
                    -raw_phase_deg / (360.0 * burst.frequency_hz) * 1000.0
                ),
                "telemetry_torque_gain": float(abs(telemetry_ratio)),
                "telemetry_torque_phase_deg": float(
                    np.angle(telemetry_ratio, deg=True)
                ),
                "raw_velocity_gain_rad_s_per_nm": float(abs(velocity_ratio)),
                "raw_velocity_phase_deg": float(
                    np.angle(velocity_ratio, deg=True)
                ),
                "excitation_velocity_gain_rad_s_per_nm": float(
                    abs(excitation_velocity_ratio)
                ),
                "excitation_velocity_phase_deg": float(
                    np.angle(excitation_velocity_ratio, deg=True)
                ),
                "reported_velocity_vs_position_derivative_gain": float(
                    abs(velocity_vs_position_derivative)
                ),
                "reported_velocity_vs_position_derivative_phase_deg": float(
                    np.angle(velocity_vs_position_derivative, deg=True)
                ),
                "acceleration_gain_rad_s2_per_nm": float(abs(acceleration_ratio)),
                "acceleration_phase_deg": float(
                    np.angle(acceleration_ratio, deg=True)
                ),
                "acceleration_vs_feedback_phase_deg": float(
                    np.angle(acceleration_phasor / feedback_phasor, deg=True)
                ),
            }
        )

    if len(phases) >= 3:
        frequency_array = np.asarray(frequencies)
        phase_radians = np.unwrap(np.deg2rad(phases))
        slope, intercept = np.polyfit(frequency_array, phase_radians, 1)
        predicted = slope * frequency_array + intercept
        residual = np.sum(np.square(phase_radians - predicted))
        total = np.sum(np.square(phase_radians - np.mean(phase_radians)))
        result["phase_fit"] = {
            "pure_delay_ms": float(-slope / (2.0 * np.pi) * 1000.0),
            "phase_intercept_deg": float(np.rad2deg(intercept)),
            "r_squared": float(1.0 - residual / total) if total > 0.0 else 0.0,
        }
        result["first_order_lag_plus_delay_fit"] = fit_first_order_lag_plus_delay(
            frequencies, raw_responses
        )
        result["reported_velocity_filter_fit"] = fit_first_order_lag_plus_delay(
            frequencies, reported_velocity_filter_responses
        )
    return result


def write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def write_raw_can_csv(
    path,
    frames,
    experiment_start_ns,
    target_motor_id,
    torque_constant,
    velocity_feedback_scale=1.0,
):
    fields = [
        "elapsed_s", "kernel_realtime_ns", "kernel_monotonic_ns",
        "host_receive_monotonic_ns", "can_id", "dlc", "data_hex",
        "message_flags", "locally_generated", "target_frame_kind",
        "decoded_torque_raw", "decoded_torque_motor_units", "decoded_torque_sdk_nm",
        "decoded_kp", "decoded_kd", "decoded_position", "decoded_velocity",
        "decoded_velocity_sdk_rad_s", "decoded_error",
    ]
    rows = []
    for frame in frames:
        kind = ""
        decoded = None
        if frame.dlc == 8 and frame.locally_generated and frame.can_id == target_motor_id:
            kind = "command"
            decoded = decode_dm_command(frame.data)
        elif frame.dlc == 8 and not frame.locally_generated:
            candidate = decode_dm_feedback(frame.data)
            if candidate["motor_id"] == target_motor_id:
                kind = "feedback"
                decoded = candidate
        torque = decoded.get("torque") if decoded else None
        rows.append([
            (frame.kernel_monotonic_ns - experiment_start_ns) * 1e-9,
            frame.kernel_realtime_ns,
            frame.kernel_monotonic_ns,
            frame.host_receive_monotonic_ns,
            frame.can_id,
            frame.dlc,
            frame.data.hex(),
            frame.message_flags,
            int(frame.locally_generated),
            kind,
            decoded.get("torque_raw") if decoded else "",
            torque if torque is not None else "",
            torque * torque_constant if torque is not None else "",
            decoded.get("kp", "") if decoded else "",
            decoded.get("kd", "") if decoded else "",
            decoded.get("position", "") if decoded else "",
            decoded.get("velocity", "") if decoded else "",
            (
                decoded.get("velocity") * velocity_feedback_scale
                if decoded and kind == "feedback"
                else ""
            ),
            decoded.get("error", "") if decoded else "",
        ])
    write_csv(path, fields, rows)


def write_html_report(path, columns, rows, frames, metadata):
    data = np.asarray(rows, dtype=np.float64)
    ci = {name: index for index, name in enumerate(columns)}
    elapsed = data[:, ci["host_elapsed_s"]]
    joint = metadata["joint"]
    plots = [
        _svg_plot(
            "Target-joint torque",
            elapsed,
            [
                ("profile", data[:, ci["profile_torque_nm"]]),
                (
                    "host damping",
                    data[:, ci[f"host_joint_damping_torque_{joint}"]],
                ),
                ("final host command", data[:, ci[f"joint_torque_command_{joint}"]]),
                ("returned torque", data[:, ci[f"joint_torque_measured_{joint}"]]),
            ],
        ),
        _svg_plot(
            "Target-joint motion",
            elapsed,
            [
                ("position (rad)", data[:, ci[f"joint_position_{joint}"]]),
                ("velocity (rad/s)", data[:, ci[f"joint_velocity_{joint}"]]),
            ],
        ),
    ]
    frequency_rows = "".join(
        "<tr>"
        f"<td>{item['frequency_hz']:.1f}</td>"
        f"<td>{item['raw_torque_gain']:.3f}</td>"
        f"<td>{item['raw_feedback_amplitude_counts']:.1f}</td>"
        f"<td>{item['raw_torque_phase_deg']:.1f}</td>"
        f"<td>{item['raw_torque_equivalent_delay_ms']:.2f}</td>"
        f"<td>{item['raw_velocity_gain_rad_s_per_nm']:.3f}</td>"
        f"<td>{item['raw_velocity_phase_deg']:.1f}</td>"
        f"<td>{item['reported_velocity_vs_position_derivative_phase_deg']:.1f}</td>"
        f"<td>{item['acceleration_phase_deg']:.1f}</td>"
        f"<td>{item['acceleration_vs_feedback_phase_deg']:.1f}</td>"
        "</tr>"
        for item in metadata["analysis"]["frequency_response"]
    )
    dynamic_fit = metadata["analysis"].get("first_order_lag_plus_delay_fit")
    if dynamic_fit:
        fit_summary = (
            "<h2>Torque response model</h2>"
            "<p>Best first-order-lag-plus-delay fit: "
            f"DC gain {dynamic_fit['dc_gain']:.3f}, "
            f"cutoff {dynamic_fit['cutoff_frequency_hz']:.2f} Hz "
            f"(time constant {dynamic_fit['time_constant_ms']:.2f} ms), "
            f"plus {dynamic_fit['additional_delay_ms']:.2f} ms additional delay. "
            "This response includes both torque realization and the motor's "
            "returned-torque reporting path.</p>"
        )
    else:
        fit_summary = ""
    velocity_fit = metadata["analysis"].get("reported_velocity_filter_fit")
    if velocity_fit:
        velocity_fit_summary = (
            "<h2>Reported velocity model</h2>"
            "<p>Relative to the derivative of position from the same raw CAN "
            "replies: "
            f"DC gain {velocity_fit['dc_gain']:.3f}, "
            f"cutoff {velocity_fit['cutoff_frequency_hz']:.2f} Hz "
            f"(time constant {velocity_fit['time_constant_ms']:.2f} ms), "
            f"plus {velocity_fit['additional_delay_ms']:.2f} ms additional delay.</p>"
        )
    else:
        velocity_fit_summary = ""
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Torque latency recording</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#172033}}
table{{border-collapse:collapse}}th,td{{border:1px solid #ccd3dd;padding:5px 9px;text-align:right}}
svg{{width:100%;max-width:1100px;border:1px solid #e2e5eb;margin:8px 0}}
.plot-bg{{fill:#fff}}.grid{{stroke:#e5e7eb;stroke-width:1}}.title{{font-size:14px;font-weight:600}}
.tick,.legend{{font-size:10px;fill:#475569}}pre{{background:#f6f8fa;padding:12px;overflow:auto}}
</style></head><body><h1>Torque/current latency recording</h1>
<p>Joint {joint}; {len(rows)} telemetry samples; {len(frames)} raw CAN frames.</p>
{''.join(plots)}
{fit_summary}
{velocity_fit_summary}
<h2>Frequency response</h2><table><thead><tr><th>Hz</th><th>T gain</th><th>response counts</th>
<th>T phase °</th><th>equiv. delay ms</th><th>V gain rad/s/Nm</th><th>V phase °</th>
<th>V/(jωQ) phase °</th><th>accel phase °</th><th>accel−T phase °</th></tr></thead>
<tbody>{frequency_rows}</tbody></table>
<h2>Metadata and analysis</h2><pre>{html.escape(json.dumps(metadata, indent=2))}</pre>
</body></html>"""
    path.write_text(document)


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option("--joint", type=click.IntRange(1, 6), default=6, show_default=True)
@click.option(
    "--amplitude",
    type=click.FloatRange(min=0.001, max=0.5),
    default=DEFAULT_AMPLITUDE_NM,
    show_default=True,
    help="Peak SDK joint feedforward torque in Nm.",
)
@click.option(
    "--frequency",
    "frequencies",
    multiple=True,
    type=click.FloatRange(min=1.0, max=40.0),
    help="Burst frequency in Hz; repeat for multiple values.",
)
@click.option(
    "--target-motor-kd",
    type=click.FloatRange(min=0.0, max=5.0),
    default=0.0,
    show_default=True,
    help=(
        "Local DaMiao Kd on the selected joint. Record separate Kd=0 and "
        "Kd>0 runs for A/B identification."
    ),
)
@click.option(
    "--ab-motor-kd",
    type=click.FloatRange(min=0.01, max=5.0),
    default=None,
    help=(
        "Run a same-session A/B test: this damped Kd condition first, then "
        "Kd=0 at the same pose, followed by one return home."
    ),
)
@click.option(
    "--local-vs-host-motor-kd",
    type=click.FloatRange(min=0.01, max=5.0),
    default=None,
    help=(
        "Run local motor Kd first, then matched host-computed J6 damping "
        "in the SDK control loop, at the same pose."
    ),
)
@click.option(
    "--host-joint-damping",
    type=click.FloatRange(min=0.01, max=10.0),
    default=1.0,
    show_default=True,
    help="Host damping coefficient B in -B*qdot, in SDK Nm*s/rad.",
)
@click.option("--travel-limit", type=click.FloatRange(min=0.02), default=DEFAULT_TRAVEL_LIMIT_RAD, show_default=True)
@click.option("--speed-limit", type=click.FloatRange(min=0.1), default=DEFAULT_SPEED_LIMIT_RAD_S, show_default=True)
@click.option("--joint-limit-margin", type=click.FloatRange(min=0.02), default=DEFAULT_JOINT_LIMIT_MARGIN_RAD, show_default=True)
@click.option("--sample-dt", type=click.FloatRange(min=0.002), default=0.002, show_default=True)
@click.option("--output-prefix", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--skip-positioning", is_flag=True)
@click.option("--yes", "assume_yes", is_flag=True)
@click.option("--return-home/--no-return-home", default=True, show_default=True)
def main(
    model,
    interface,
    joint,
    amplitude,
    frequencies,
    target_motor_kd,
    ab_motor_kd,
    local_vs_host_motor_kd,
    host_joint_damping,
    travel_limit,
    speed_limit,
    joint_limit_margin,
    sample_dt,
    output_prefix,
    skip_positioning,
    assume_yes,
    return_home,
):
    """Record a small-signal t_ff latency test on a DaMiao joint."""
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    if robot_config.joint_dof != 6:
        raise click.ClickException("This recorder currently requires a six-joint model.")
    target_index = joint - 1
    target_motor_id = int(robot_config.motor_id[target_index])
    velocity_feedback_scale = float(
        robot_config.joint_velocity_feedback_scale[target_index]
    )
    try:
        torque_constant = motor_torque_constant(robot_config.motor_type[target_index])
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config
    )
    if sample_dt < controller_config.controller_dt - 1e-12:
        raise click.ClickException(
            f"--sample-dt cannot be below the {controller_config.controller_dt:g} s controller period"
        )
    if (
        ab_motor_kd is not None or local_vs_host_motor_kd is not None
    ) and abs(target_motor_kd) > 1e-12:
        raise click.ClickException(
            "--target-motor-kd must remain 0 in a same-session comparison"
        )
    if ab_motor_kd is not None and local_vs_host_motor_kd is not None:
        raise click.ClickException(
            "use either --ab-motor-kd or --local-vs-host-motor-kd, not both"
        )
    selected_frequencies = frequencies or DEFAULT_FREQUENCIES_HZ
    profile = TorqueBurstProfile(amplitude, selected_frequencies)
    if output_prefix is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_kd = (
            local_vs_host_motor_kd
            if local_vs_host_motor_kd is not None
            else ab_motor_kd if ab_motor_kd is not None else target_motor_kd
        )
        kd_label = f"{output_kd:g}".replace(".", "p")
        mode_label = (
            "local_vs_host"
            if local_vs_host_motor_kd is not None
            else "ab" if ab_motor_kd is not None else "kd"
        )
        output_prefix = (
            Path.cwd()
            / f"torque_latency_{model}_j{joint}_{mode_label}{kd_label}_{stamp}"
        )
    output_prefix = output_prefix.resolve()

    controller_config.background_send_recv = True
    controller_config.gravity_compensation = True
    controller_config.friction_compensation = False
    controller_config.joint_friction_compensation = False
    controller_config.gripper_friction_compensation = False
    controller = None
    recorder = None
    all_frames = []
    condition_records = []
    failure = None
    returned_home = False
    columns = telemetry_columns()
    condition_specs = motor_kd_condition_specs(
        target_motor_kd,
        ab_motor_kd,
        local_vs_host_motor_kd,
        host_joint_damping,
    )

    try:
        controller = arx5.Arx5JointController(
            robot_config, controller_config, interface
        )
        controller.set_log_level(arx5.LogLevel.WARNING)
        controller.set_gravity_compensation(True)
        controller.set_friction_compensation(False)
        controller.set_to_damping()
        if not skip_positioning:
            while True:
                click.prompt(
                    "Move the gravity-compensated, damped arm to a clear test pose; release it and press Enter",
                    default="",
                    show_default=False,
                )
                maximum_speed = 0.0
                stationary_start = time.monotonic()
                while time.monotonic() - stationary_start < 0.25:
                    velocity = np.asarray(
                        controller.get_joint_state().vel(), dtype=np.float64
                    )
                    maximum_speed = max(maximum_speed, float(np.max(np.abs(velocity))))
                    time.sleep(0.01)
                if maximum_speed <= 0.10:
                    break
                click.echo(
                    f"Arm still moving (peak {maximum_speed:.3f} rad/s); try again.",
                    err=True,
                )

        initial_anchor_state = controller.get_joint_state()
        initial_anchor_position = np.asarray(
            initial_anchor_state.pos(), dtype=np.float64
        ).copy()
        initial_clearance = np.minimum(
            initial_anchor_position - np.asarray(robot_config.joint_pos_min),
            np.asarray(robot_config.joint_pos_max) - initial_anchor_position,
        )
        if np.min(initial_clearance) <= joint_limit_margin:
            raise click.ClickException("Start pose is too close to a joint limit.")
        if local_vs_host_motor_kd is not None:
            confirmation = (
                f"Run same-session local-vs-host bursts of ±{amplitude:g} Nm "
                f"from {min(selected_frequencies):g} to "
                f"{max(selected_frequencies):g} Hz on joint {joint}: motor-local "
                f"Kd={local_vs_host_motor_kd:g} first, then motor Kd=0 with "
                f"host damping B={host_joint_damping:g} Nm*s/rad? "
            )
        elif ab_motor_kd is not None:
            confirmation = (
                f"Run same-session A/B bursts of ±{amplitude:g} Nm from "
                f"{min(selected_frequencies):g} to {max(selected_frequencies):g} Hz "
                f"on joint {joint}, with motor Kp=0, Kd={ab_motor_kd:g} first, "
                "then Kd=0 without releasing the controller? "
            )
        else:
            confirmation = (
                f"Apply ±{amplitude:g} Nm sine bursts from "
                f"{min(selected_frequencies):g} to {max(selected_frequencies):g} Hz "
                f"on joint {joint} with motor Kp=0 and Kd={target_motor_kd:g}? "
            )
        if not assume_yes and not click.confirm(
            confirmation + f"Clear possible ±{travel_limit:g} rad motion first."
        ):
            raise click.Abort()

        recorder = RawCanRecorder(interface)
        recorder.start()
        sample_period_ns = int(round(sample_dt * 1e9))
        first_condition_anchor = None

        for condition_index, (
            condition_label,
            condition_kd,
            condition_host_damping,
        ) in enumerate(
            condition_specs
        ):
            if condition_index > 0:
                peak_settling_speed = wait_for_stationary_arm(controller)
                click.echo(
                    f"First condition settled (observed peak during settling "
                    f"{peak_settling_speed:.3f} rad/s)."
                )

            anchor_state = controller.get_joint_state()
            anchor_position = np.asarray(
                anchor_state.pos(), dtype=np.float64
            ).copy()
            if first_condition_anchor is None:
                first_condition_anchor = anchor_position.copy()
                anchor_delta = 0.0
            else:
                anchor_delta = float(
                    np.max(np.abs(anchor_position - first_condition_anchor))
                )
                if anchor_delta > AB_MAX_ANCHOR_DELTA_RAD:
                    raise RuntimeError(
                        "A/B pose drifted by "
                        f"{anchor_delta:.3f} rad before the second condition; "
                        "comparison was not run"
                    )

            condition_gain = arx5.Gain(robot_config.joint_dof)
            condition_gain.kp()[:] = 0.0
            condition_gain.kd()[:] = np.asarray(controller_config.default_kd)
            condition_gain.kd()[target_index] = condition_kd
            condition_gain.gripper_kp = controller_config.default_gripper_kp
            condition_gain.gripper_kd = controller_config.default_gripper_kd
            host_damping_vector = np.zeros(
                robot_config.joint_dof, dtype=np.float64
            )
            if local_vs_host_motor_kd is not None:
                host_damping_vector[target_index] = condition_host_damping
                controller._set_host_joint_damping(host_damping_vector)
            controller.set_gain(condition_gain)

            condition_rows = []
            condition_failure = None
            experiment_start_ns = time.monotonic_ns()
            next_sample_ns = experiment_start_ns
            click.echo(
                f"Recording {condition_label} condition {condition_index + 1}/"
                f"{len(condition_specs)} for {profile.duration_s:.2f} s: "
                f"joint {joint}, motor Kd={condition_kd:g}, host B="
                f"{condition_host_damping:g} Nm*s/rad. Ctrl-C returns immediately "
                "to damping."
            )

            while True:
                now_ns = time.monotonic_ns()
                if now_ns < next_sample_ns:
                    time.sleep((next_sample_ns - now_ns) * 1e-9)
                    now_ns = time.monotonic_ns()
                elapsed_s = (now_ns - experiment_start_ns) * 1e-9
                if elapsed_s >= profile.duration_s:
                    break
                torque, frequency, envelope = profile.sample(elapsed_s)
                command = arx5.JointState(
                    anchor_position,
                    np.zeros(6, dtype=np.float64),
                    np.eye(1, 6, target_index, dtype=np.float64).reshape(6)
                    * torque,
                    anchor_state.gripper_pos,
                )
                command.timestamp = controller.get_timestamp()
                call_start = time.monotonic()
                controller.set_joint_cmd(command)
                call_duration = time.monotonic() - call_start
                condition_rows.append(
                    capture_telemetry_row(
                        controller,
                        elapsed_s,
                        now_ns,
                        torque,
                        frequency,
                        envelope,
                        call_duration,
                        host_damping_vector,
                    )
                )
                state = controller.get_joint_state()
                reason = safety_guard_reason(
                    state.pos(),
                    state.vel(),
                    anchor_position,
                    np.asarray(robot_config.joint_pos_min),
                    np.asarray(robot_config.joint_pos_max),
                    target_index,
                    travel_limit,
                    speed_limit,
                    joint_limit_margin,
                )
                if reason:
                    condition_failure = f"safety guard: {reason}"
                    failure = condition_failure
                    break
                skipped = max(
                    0, (now_ns - next_sample_ns) // sample_period_ns
                )
                next_sample_ns += (skipped + 1) * sample_period_ns

            experiment_end_ns = time.monotonic_ns()
            condition_records.append(
                {
                    "label": condition_label,
                    "kd": condition_kd,
                    "host_damping": condition_host_damping,
                    "rows": condition_rows,
                    "start_ns": experiment_start_ns,
                    "end_ns": experiment_end_ns,
                    "anchor_position": anchor_position,
                    "anchor_gripper_position": float(anchor_state.gripper_pos),
                    "anchor_delta_from_first_rad": anchor_delta,
                    "failure": condition_failure,
                }
            )
            if failure:
                break

        if local_vs_host_motor_kd is not None:
            controller._set_host_joint_damping(
                np.zeros(robot_config.joint_dof, dtype=np.float64)
            )
        controller.set_to_damping()
        time.sleep(0.10)
        all_frames = recorder.stop()
        recorder = None
        if failure is None and return_home:
            click.echo("All latency conditions complete; returning home.")
            controller.reset_to_home()
            time.sleep(HOME_FINAL_SETTLE_S)
            returned_home = True
    except KeyboardInterrupt:
        failure = "operator interrupted recording"
        click.echo("Interrupted; returning immediately to damping.", err=True)
    except click.Abort:
        failure = "operator canceled before recording"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        click.echo(f"Latency test stopped: {failure}", err=True)
    finally:
        if controller is not None:
            try:
                if local_vs_host_motor_kd is not None:
                    controller._set_host_joint_damping(
                        np.zeros(robot_config.joint_dof, dtype=np.float64)
                    )
                controller.set_to_damping()
            except Exception as exc:
                failure = failure or f"failed to restore damping: {exc}"
        if recorder is not None:
            try:
                all_frames = recorder.stop()
            except Exception as exc:
                failure = failure or str(exc)

    if not condition_records or not any(
        record["rows"] for record in condition_records
    ):
        if failure == "operator canceled before recording":
            raise click.Abort()
        raise click.ClickException(failure or "No telemetry samples were recorded.")

    metadata_by_label = {}
    for record in condition_records:
        condition_frames = frames_in_monotonic_window(
            all_frames, record["start_ns"], record["end_ns"]
        )
        analysis = analyze_latency(
            profile,
            record["rows"],
            condition_frames,
            target_index,
            target_motor_id,
            torque_constant,
            velocity_feedback_scale,
            record["start_ns"],
        )
        metadata = {
            "model": model,
            "interface": interface,
            "joint": joint,
            "motor_id": target_motor_id,
            "torque_constant": torque_constant,
            "joint_velocity_feedback_scale": velocity_feedback_scale,
            "amplitude_nm": amplitude,
            "frequencies_hz": list(selected_frequencies),
            "profile_duration_s": profile.duration_s,
            "profile_ramp_duration_s": profile.ramp_duration_s,
            "profile_gap_duration_s": profile.gap_duration_s,
            "profile_lead_duration_s": profile.lead_duration_s,
            "anchor_joint_position_rad": record["anchor_position"].tolist(),
            "anchor_gripper_position": record["anchor_gripper_position"],
            "anchor_delta_from_first_condition_rad": record[
                "anchor_delta_from_first_rad"
            ],
            "session_condition_label": record["label"],
            "session_condition_order": [item[0] for item in condition_specs],
            "selected_joint_motor_kp": 0.0,
            "selected_joint_motor_kd": record["kd"],
            "host_joint_damping_nm_s_per_rad": record["host_damping"],
            "assumed_effective_joint_damping_nm_s_per_rad": (
                torque_constant * record["kd"]
            ),
            "effective_damping_note": (
                "Assumes DaMiao Kd is motor torque-unit per physical rad/s; "
                "its positive scale affects inferred magnitude but not "
                "inferred phase."
            ),
            "other_joint_motor_kd": np.asarray(
                controller_config.default_kd
            ).tolist(),
            "travel_limit_rad": travel_limit,
            "speed_limit_rad_s": speed_limit,
            "failure": record["failure"],
            "returned_home": returned_home,
            "raw_timestamp_semantics": {
                "kernel_realtime_ns": "Linux SO_TIMESTAMPNS software timestamp",
                "kernel_monotonic_ns": "realtime timestamp mapped once to time.monotonic_ns",
                "outgoing_loopback": "kernel local-loopback/enqueue time, not a hardware on-wire TX timestamp",
                "incoming": "kernel software receive timestamp, not the motor sensor-sampling time",
            },
            "analysis": analysis,
        }
        if ab_motor_kd is None and local_vs_host_motor_kd is None:
            condition_prefix = output_prefix
        else:
            kd_label = f"{record['kd']:g}".replace(".", "p")
            condition_prefix = output_prefix.with_name(
                f"{output_prefix.name}_{record['label']}_kd{kd_label}"
            )
        telemetry_path = condition_prefix.with_suffix(".csv")
        raw_path = condition_prefix.with_name(condition_prefix.name + "_can.csv")
        json_path = condition_prefix.with_suffix(".json")
        html_path = condition_prefix.with_suffix(".html")
        write_csv(telemetry_path, columns, record["rows"])
        write_raw_can_csv(
            raw_path,
            condition_frames,
            record["start_ns"],
            target_motor_id,
            torque_constant,
            velocity_feedback_scale,
        )
        json_path.write_text(json.dumps(metadata, indent=2) + "\n")
        write_html_report(
            html_path, columns, record["rows"], condition_frames, metadata
        )
        metadata_by_label[record["label"]] = metadata
        click.echo(f"{record['label'].title()} telemetry: {telemetry_path}")
        click.echo(f"{record['label'].title()} raw CAN:   {raw_path}")
        click.echo(f"{record['label'].title()} report:    {html_path}")
        click.echo(f"{record['label'].title()} meta:      {json_path}")

    if ab_motor_kd is not None and failure is None:
        comparison = compare_motor_kd_frequency_response(
            metadata_by_label["baseline"], metadata_by_label["damped"]
        )
        comparison_prefix = output_prefix.with_name(
            output_prefix.name + "_comparison"
        )
        comparison_json = comparison_prefix.with_suffix(".json")
        comparison_csv = comparison_prefix.with_suffix(".csv")
        comparison_html = comparison_prefix.with_suffix(".html")
        comparison_json.write_text(json.dumps(comparison, indent=2) + "\n")
        from compare_torque_latency_ab import (
            write_comparison_csv,
            write_comparison_html,
        )

        write_comparison_csv(comparison_csv, comparison)
        write_comparison_html(comparison_html, comparison)
        summary = comparison["summary"]
        click.echo(
            "A/B closer phase hypothesis: "
            f"{summary['closer_phase_hypothesis']}"
        )
        click.echo(f"A/B CSV:    {comparison_csv}")
        click.echo(f"A/B report: {comparison_html}")
        click.echo(f"A/B meta:   {comparison_json}")
    elif local_vs_host_motor_kd is not None and failure is None:
        comparison = compare_local_vs_host_damping(
            metadata_by_label["local"], metadata_by_label["host"]
        )
        comparison_prefix = output_prefix.with_name(
            output_prefix.name + "_comparison"
        )
        comparison_json = comparison_prefix.with_suffix(".json")
        comparison_csv = comparison_prefix.with_suffix(".csv")
        comparison_html = comparison_prefix.with_suffix(".html")
        comparison_json.write_text(json.dumps(comparison, indent=2) + "\n")
        from compare_torque_latency_ab import (
            write_comparison_csv,
            write_local_host_comparison_html,
        )

        write_comparison_csv(comparison_csv, comparison)
        write_local_host_comparison_html(comparison_html, comparison)
        summary = comparison["summary"]
        click.echo(
            "Local-vs-host median velocity gain ratio: "
            f"{summary['median_host_over_local_gain']:.3f}; host-minus-local "
            f"phase {summary['median_host_minus_local_phase_deg']:.1f} deg."
        )
        click.echo(f"Comparison CSV:    {comparison_csv}")
        click.echo(f"Comparison report: {comparison_html}")
        click.echo(f"Comparison meta:   {comparison_json}")
    if failure:
        raise click.ClickException(failure)


if __name__ == "__main__":
    main()
