
# Support direct execution as well as package imports.
if __package__ in (None, ''):
    import sys as _path_sys
    from pathlib import Path as _Path
    _path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from contracts.action_contract import require_absolute_actions
from config.supported_policy import require_supported_policy, require_unmasked_for_serving
from contracts.action_rate_contract import resolve_policy_rate
import argparse
import asyncio
import http
import json
import logging
import os
import pickle
import socket
import time
import traceback
from typing import Any

import cv2
from contracts import standard_image_processing
import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from policy.model_util import make_policy
from inference.inference_utils import fill_in


POLICY_FAMILY = "flash_aloha"
LOCAL_POLICY_REQUEST_SCHEMA = "native"
SUPPORTED_POLICY_CLASSES = {"Flow", "ACT"}


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def msgpack_unpackb(data):
    return msgpack.unpackb(data, object_hook=unpack_array)


class LocalPolicyServer:
    def __init__(
        self,
        *,
        policy,
        stats: dict[str, Any],
        config: dict[str, Any],
        ckpt_path: str,
        host: str,
        port: int,
        device: torch.device,
        use_fill_in: bool,
    ):
        require_absolute_actions(config)
        standard_image_processing.require_single_frame(config)
        self.policy = policy
        self.stats = stats
        self.config = config
        require_supported_policy(config)
        require_unmasked_for_serving(config)
        standard_image_processing.policy_image_spec(config)
        self.policy_config = config["policy_config"]
        from config.task_config import checkpoint_chunk_size
        self.chunk_size = checkpoint_chunk_size(config)  # the training config's model.chunk_size
        self.policy_class = config["policy_class"]
        self.ckpt_path = ckpt_path
        self.host = host
        self.port = port
        self.device = device
        self.use_fill_in = use_fill_in
        self.packer = msgpack.Packer(default=pack_array)
        self.camera_names = list(self.policy_config.get("camera_names", []))
        self.height = int(self.policy_config.get("height", config.get("height")))
        self.width = int(self.policy_config.get("width", config.get("width")))
        self.state_dim = int(self.policy_config["state_dim"])
        self.action_dim = int(self.policy_config["action_dim"])
        from contracts.robot_contract import metadata_robot_contract, require_robot_match
        robot = metadata_robot_contract(config)
        if robot is not None:
            require_robot_match(robot, stats, context='policy normalization statistics')
        elif metadata_robot_contract(stats) is not None:
            raise ValueError('Tagged normalization statistics require a checkpoint robot_contract')
        self.metadata = self._build_metadata()

    def _build_metadata(self) -> dict[str, Any]:
        from camera_utils.camera_settings_contract import metadata_camera_settings
        from config.rtc_config import checkpoint_rtc
        from contracts.robot_contract import metadata_robot_contract
        robot = metadata_robot_contract(self.config)
        return {
            **({'robot_contract': robot} if robot is not None else {}),
            **checkpoint_rtc(self.config),
            "camera_settings": metadata_camera_settings(self.config),
            "prediction_horizon": self.chunk_size,
            "base_camera_names": self.policy_config.get("base_camera_names", self.camera_names),
            "action_command_style": "absolute",
            "control_freq": resolve_policy_rate(self.config),
            "policy_family": POLICY_FAMILY,
            "policy_class": self.policy_class,
            "request_schema": LOCAL_POLICY_REQUEST_SCHEMA,
            "rtc_modes": ["training_time"] if self.policy_class == "Flow" else [],
            "camera_names": self.camera_names,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "robot_action_dim": self.action_dim,
            "arm_count": 2 if self.action_dim > 7 else 1,
            "is_double_arm": self.action_dim > 7,
            "height": self.height,
            "width": self.width,
            "ckpt_path": self.ckpt_path,
            "ckpt_dir": self.config.get("ckpt_dir"),
            "task_name": self.config.get("task_name") or self.config.get("task_config", {}).get("task_name"),
            "single_arm_side": self.policy_config.get("single_arm_side"),
            "single_arm_can": self.policy_config.get("single_arm_can"),
        }

    def serve_forever(self) -> None:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info(
            "Serving %s policy on %s:%s (host=%s, ip=%s)",
            self.policy_class,
            self.host,
            self.port,
            hostname,
            local_ip,
        )
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        logging.info("Connection from %s opened", websocket.remote_address)
        await websocket.send(self.packer.pack(self.metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_unpackb(await websocket.recv())

                infer_start = time.monotonic()
                response = self.infer(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
                response["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(self.packer.pack(response))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    @torch.inference_mode()
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        qpos = self._process_qpos(obs)
        image = self._process_images(obs)
        from config.task_config import require_chunk_length
        actions = require_chunk_length(self._infer_policy_actions(qpos, image, obs).detach().cpu().numpy(),
                                       self.chunk_size)
        actions = actions * self.stats["action_std"] + self.stats["action_mean"]
        if self.use_fill_in:
            actions = fill_in(actions)
        return {"actions": np.asarray(actions, dtype=np.float32)}

    def _infer_policy_actions(self, qpos: torch.Tensor, image: torch.Tensor, obs: dict[str, Any]) -> torch.Tensor:
        rtc_prefix = self._build_training_time_rtc_prefix(obs)
        if rtc_prefix is None:
            return self.policy.forward_inf(qpos, image)
        if hasattr(self.policy, "forward_inf_training_time_rtc"):
            return self.policy.forward_inf_training_time_rtc(qpos, image, rtc_prefix)
        raise RuntimeError(f"{self.policy_class} RTC requires policy.forward_inf_training_time_rtc")

    def _build_training_time_rtc_prefix(self, obs: dict[str, Any]) -> torch.Tensor | None:
        if not _coerce_bool(obs.get("rtc/enabled", False)):
            return None
        if "rtc/prev_chunk_left_over" not in obs:
            return None
        inference_delay_steps = _coerce_int(obs.get("rtc/inference_delay_steps", 0))
        if inference_delay_steps <= 0:
            return None

        prev_chunk_left_over = np.asarray(obs["rtc/prev_chunk_left_over"], dtype=np.float32)
        if prev_chunk_left_over.ndim != 2:
            raise ValueError(
                "rtc/prev_chunk_left_over must be a 2-D action chunk, "
                f"got shape {prev_chunk_left_over.shape}"
            )
        if prev_chunk_left_over.shape[1] != self.action_dim:
            raise ValueError(
                f"rtc/prev_chunk_left_over action dim {prev_chunk_left_over.shape[1]} "
                f"does not match {self.policy_class} action_dim {self.action_dim}"
            )
        prefix_len = min(int(inference_delay_steps), prev_chunk_left_over.shape[0])
        if prefix_len <= 0:
            return None

        prefix = prev_chunk_left_over[:prefix_len]
        prefix = (prefix - self.stats["action_mean"]) / self.stats["action_std"]
        return torch.from_numpy(prefix).float().unsqueeze(0).to(self.device)

    def _process_qpos(self, obs: dict[str, Any]) -> torch.Tensor:
        if "observation/state" not in obs:
            raise KeyError("Flow request missing observation/state")
        qpos = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)
        if qpos.shape[0] != self.state_dim:
            raise ValueError(f"Expected observation/state dim {self.state_dim}, got {qpos.shape[0]}")
        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return torch.from_numpy(qpos).float().unsqueeze(0).to(self.device)

    def _process_images(self, obs: dict[str, Any]) -> torch.Tensor:
        color_space = _decode_text(obs.get("client/image_color_space", "BGR")).upper()
        images = []
        for camera_name in self.camera_names:
            image = self._lookup_image(obs, camera_name)
            image = self._prepare_image(image, color_space=color_space, camera_name=camera_name)
            images.append(image)
        spec = standard_image_processing.policy_image_spec(self.config)
        tensor = torch.from_numpy(np.stack([
            standard_image_processing.image_to_chw(image, standard_image_processing.ImageSpec(
                color=spec.color, grayscale=spec.grayscale), source_color=spec.color)
            for image in images
        ])).unsqueeze(0)
        return tensor.to(self.device)

    def _lookup_image(self, obs: dict[str, Any], camera_name: str) -> np.ndarray:
        exact_key = f"observation/images/{camera_name}"
        if exact_key in obs:
            return np.asarray(obs[exact_key])

        aliases = {
            "video_overhead": ("observation/image",),
            "video_left": ("observation/wrist_image",),
            "video_right": ("observation/wrist_image_right",),
        }
        for key in aliases.get(camera_name, ()):
            if key in obs:
                return np.asarray(obs[key])
        raise KeyError(f"Flow request missing image for camera {camera_name!r} ({exact_key})")

    def _prepare_image(self, image: np.ndarray, *, color_space: str, camera_name: str) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Camera {camera_name} expected HxWx3 image, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = np.asarray(image)
            if np.issubdtype(image.dtype, np.floating) and image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return standard_image_processing.prepare_image(
            image, standard_image_processing.policy_image_spec(self.config),
            source_color=color_space)



def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return False


def _coerce_int(value: Any) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return int(value)


def _health_check(connection: websocket_server.ServerConnection, request: websocket_server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _select_ckpt(config: dict[str, Any], *, policy_step: int, ckpt_name: str | None) -> str:
    ckpt_dir = config["ckpt_dir"]
    if ckpt_name is None:
        if policy_step < 0:
            ckpt_name = "policy_best.ckpt"
            if not os.path.isfile(os.path.join(ckpt_dir, ckpt_name)):
                ckpt_name = "policy_last.ckpt"
        else:
            ckpt_name = f"policy_step_{policy_step}_seed_0.ckpt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _validate_config_and_stats(config: dict[str, Any], stats: dict[str, Any]) -> None:
    require_absolute_actions(config)
    require_supported_policy(config)
    require_unmasked_for_serving(config)
    policy_class = config.get("policy_class")
    if policy_class not in SUPPORTED_POLICY_CLASSES:
        supported = ", ".join(sorted(SUPPORTED_POLICY_CLASSES))
        raise ValueError(
            f"serve_policy.py only supports policy_class in {{{supported}}}, got {policy_class!r}"
        )
    policy_config = config.get("policy_config") or {}
    required_policy_keys = ("camera_names", "state_dim", "action_dim", "height", "width")
    missing = [key for key in required_policy_keys if key not in policy_config and key not in config]
    if missing:
        raise KeyError(f"{policy_class} config missing required keys: {missing}")
    state_dim = int(policy_config["state_dim"])
    action_dim = int(policy_config["action_dim"])
    for key in ("qpos_mean", "qpos_std"):
        actual = np.asarray(stats[key]).reshape(-1).shape[0]
        if actual != state_dim:
            raise ValueError(f"Stats {key} dim {actual} does not match {policy_class} model state_dim {state_dim}")
    for key in ("action_mean", "action_std"):
        actual = np.asarray(stats[key]).reshape(-1).shape[0]
        if actual != action_dim:
            raise ValueError(f"Stats {key} dim {actual} does not match {policy_class} action_dim {action_dim}")


WebsocketPolicyServer = LocalPolicyServer


def create_server(args: argparse.Namespace) -> LocalPolicyServer:
    from config.task_config import load_inference_checkpoint_config
    config = load_inference_checkpoint_config(vars(args))
    from contracts.action_rate_contract import resolve_policy_rate
    config["control_freq"] = resolve_policy_rate(config, legacy_rate=getattr(args, "legacy_policy_fps", None), context="checkpoint")
    from config.task_config import select_task_config
    task = select_task_config(config.get('task_name') or config.get('task_config', {}).get('task_name'),
                              getattr(args, 'task_config', None))
    if task:
        task.require_checkpoint(config)
    if args.fill_in:
        raise ValueError("Use native SDK interpolation; --fill_in would change the declared action rate")
    if args.diffusion_inference_steps is not None:
        if args.diffusion_inference_steps <= 0:
            raise ValueError("--diffusion_inference_steps must be > 0")
        config["policy_config"]["num_inference_timesteps"] = int(args.diffusion_inference_steps)

    stats_path = os.path.join(config["ckpt_dir"], "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    stats = {
        key: np.asarray(value, dtype=np.float32)
        if key.endswith(("mean", "std", "q01", "q99"))
        else value
        for key, value in stats.items()
    }
    _validate_config_and_stats(config, stats)

    ckpt_path = _select_ckpt(config, policy_step=args.policy_step, ckpt_name=args.ckpt_name)
    device = torch.device(args.device)
    policy = make_policy(config["policy_class"], config["policy_config"])
    loading_status = policy.deserialize(torch.load(ckpt_path, map_location=device))
    logging.info("Loaded checkpoint %s: %s", ckpt_path, loading_status)
    policy.to(device)
    policy.eval()
    if args.fast_inference:
        from policy.model_util import accelerate_policy
        accelerate_policy(policy)

    return LocalPolicyServer(
        policy=policy,
        stats=stats,
        config=config,
        ckpt_path=ckpt_path,
        host=args.host,
        port=args.port,
        device=device,
        use_fill_in=bool(args.fill_in) and not bool(args.no_fill_in),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a flash-aloha local policy over the OpenPI websocket protocol.")
    parser.add_argument("--legacy-policy-fps", type=float, default=None, help="Explicit trained action FPS for legacy checkpoints without metadata.")
    parser.add_argument("--task-config", help="Task YAML; defaults to checkpoint task name.")
    parser.add_argument("--config", "-c", default=None, help="Checkpoint all_configs.json; defaults to inference.checkpoint in task YAML.")
    parser.add_argument("--policy_step", "-p", type=int, default=-1, help="-1 uses policy_best.ckpt; otherwise policy_step_N_seed_0.ckpt.")
    parser.add_argument("--ckpt_name", type=str, default=None, help="Explicit checkpoint filename inside ckpt_dir.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--diffusion_inference_steps", type=int, default=None)
    parser.add_argument("--fast_inference", action="store_true", default=False,
                        help="Flow: fp16/bf16 channels-last U-Net convolutions and a CUDA-graphed U-Net step; "
                             "ACT: a CUDA-graphed forward.")
    parser.add_argument("--fill_in", action="store_true", default=False, help="Enable interpolation fill-in for served action chunks.")
    parser.add_argument("--no_fill_in", action="store_true", default=False, help="Deprecated compatibility flag; raw chunks are already the default.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args()
    server = create_server(args)
    logging.info("Server metadata: %s", server.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
