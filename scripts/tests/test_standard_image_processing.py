"""Standard image-processing parity. Synthetic pixels only; no hardware access."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
from data_processing import episode_store
import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from contracts import standard_image_processing as images
from contracts import standard_capture
from policy import training_utils
from inference import serve_policy


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("aloha", [False, True])
@pytest.mark.parametrize("grayscale", [False, True])
@pytest.mark.parametrize("history", [0])
def test_actual_episode_loader_matches_inference(tmp_path, compressed, aloha, grayscale, history):
    rng = np.random.default_rng(73)
    names = ["video_right", "video_overhead", "video_left"]
    raw = {name: rng.integers(0, 256, (3, 31, 47, 3), dtype=np.uint8) for name in names}
    from episode_fixtures import recorded_samples
    from data_processing.episode_store import publish_samples
    from data_processing.episode_data import Episode
    path = tmp_path / "episode.zarr"
    samples = recorded_samples([SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7),
        timestamp=i/100, images={name: frames[i] for name, frames in raw.items()})
        for i in range(3)])
    publish_samples(path, samples)
    with Episode(path) as episode:
        decoded = {name: episode.images[name][:] for name in names}
    stats = {f"{key}_{suffix}": np.full(7, value, np.float32)
             for key in ("qpos", "action") for suffix, value in (("mean", 0), ("std", 1))}
    # The masker's calibration setup is irrelevant here; its identity case is simulated.
    dataset = training_utils.EpisodicDataset(
        None, [str(path)], names, stats, np.array([0]), [3], 2, "Flow",
        width=19, height=13, observation_name=["qpos"],
        aloha=aloha, grayscale=grayscale, image_history_len=history)
    for index in (0, 2):
        current = standard_capture.resized_images(
            SimpleNamespace(images={name: decoded[name][index] for name in names}),
            {"width": 19, "height": 13})
        histories = {name: [current[name]] + [
            images.resize_image(decoded[name][index - offset], 19, 13)
            if index >= offset else np.zeros_like(current[name])
            for offset in range(1, history + 1)] for name in names}
        actual = images.inference_images(
            current, camera_names=names, aloha=aloha, grayscale=grayscale, device="cpu",
            image_history_buffer=SimpleNamespace(get_images_with_history=lambda _: histories) if history else None)
        expected = dataset[index][0].reshape(-1, 3, 13, 19).unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if not history:
            server = serve_policy.LocalPolicyServer(
                policy=None, stats=stats,
                config={"control_freq": 100, "policy_class": "Flow", "task_config": {"aloha": aloha},
                        "grayscale": grayscale, "policy_config": {
                            "camera_names": names, "height": 13, "width": 19,
                            "state_dim": 7, "action_dim": 7, "num_queries": 2}},
                ckpt_path="synthetic", host="localhost", port=0,
                device=torch.device("cpu"), use_fill_in=False)
            for color in ("RGB", "BGR"):
                obs = {"client/image_color_space": color}
                obs.update({f"observation/images/{name}": images.convert_color(decoded[name][index], "BGR", color) for name in names})
                torch.testing.assert_close(server._process_images(obs), expected, rtol=0, atol=0)




@pytest.mark.parametrize("filename", ["inference/inference_direct.py"])
def test_direct_adapters_delegate_without_hardware_imports(filename):
    tree = ast.parse((SRC / filename).read_text())
    names = {"process_image"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == 1
    delegate = mock.Mock(return_value="processed")
    namespace = {"standard_image_processing": SimpleNamespace(inference_images=delegate)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    for name in names:
        assert namespace[name]({"camera": "frame"}, grayscale=True, aloha=True) == "processed"
        assert delegate.call_args.kwargs["grayscale"] is True
        assert delegate.call_args.kwargs["aloha"] is True
    assert "from contracts import standard_image_processing" in (SRC / filename).read_text()


def test_crop_color_scale_and_no_input_mutation():
    frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    original = frame.copy()
    spec = images.ImageSpec(5, 4, crop=(1, 2, 8, 10))
    actual = images.image_to_chw(frame, spec)
    expected = cv2.resize(frame[1:9, 2:12], (5, 4), interpolation=cv2.INTER_AREA)[..., ::-1]
    np.testing.assert_array_equal(actual, expected.transpose(2, 0, 1).astype(np.float32) / 255)
    np.testing.assert_array_equal(frame, original)
    with pytest.raises(ValueError, match="outside"):
        images.prepare_image(frame, images.ImageSpec(crop=(0, 0, 99, 99)))
    with pytest.raises(ValueError, match="color"):
        images.prepare_image(frame, source_color="unknown")




def test_mask_configuration_cannot_silently_diverge():
    config = {"task_config": {"masking_strategy": "robot"}}
    assert images.masking_strategy(config) == "robot"
    assert images.masking_strategy(config, "robot") == "robot"
    with pytest.raises(ValueError, match="differs from training"):
        images.masking_strategy(config, "different")


def test_native_dimensions_are_checkpoint_owned():
    frame = np.zeros((21, 37, 3), np.uint8)
    config = {"policy_config": {"width": 17, "height": 11}}
    output = standard_capture.resized_images(SimpleNamespace(images={"cam": frame}), config)
    assert output["cam"].shape == (11, 17, 3)
    with pytest.raises(ValueError, match="conflicts"):
        standard_capture.resized_images(SimpleNamespace(images={"cam": frame}), {**config, "width": 22})


# The OpenPI trainer (src/openpi/shared/client_contract_test.py in its checkout) pins the same
# value: its resize_with_pad under OpenCV 4.11 reproduces this client's letterbox exactly.
OPENPI_720P_FINGERPRINT = "ff7888bc17fe21c2e5823bfd8e7fb415bdb1cc02dab86a8296cf455c03d56191"
OPENPI_CAMERAS = {"video_overhead": {"request_key": "observation/image", "source_shape": [720, 1280, 3]},
                  "video_right": {"request_key": "observation/wrist_image", "source_shape": [720, 1280, 3]}}


def openpi_client_image_tools():
    """OpenPI's own resize, from its checkout (OPENPI_CLIENT_SRC)."""
    import os
    if not os.environ.get("OPENPI_CLIENT_SRC"):
        pytest.skip("Set OPENPI_CLIENT_SRC to an OpenPI checkout's packages/openpi-client/src")
    source = Path(os.environ["OPENPI_CLIENT_SRC"])
    if not (source / "openpi_client/image_tools.py").exists():
        pytest.skip(f"OpenPI client source not found at {source}")
    sys.path.insert(0, str(source))
    try:
        from openpi_client import image_tools
    finally:
        sys.path.remove(str(source))
    return image_tools


def test_openpi_letterbox_geometry():
    frame = np.full((720, 1280, 3), 200, np.uint8)
    output = images.letterbox_image(frame, 448, 448)
    rows = np.nonzero(output.any(axis=(1, 2)))[0]
    assert output.shape == (448, 448, 3) and (rows[0], rows[-1]) == (98, 349)
    assert (output[98:350] == 200).all() and not output[:98].any() and not output[350:].any()


@pytest.mark.parametrize("shape", [(720, 1280, 3), (800, 1280, 3), (480, 640, 3), (333, 517, 3), (448, 448, 3)])
def test_openpi_letterbox_matches_openpi_resize_with_pad(shape):
    image_tools = openpi_client_image_tools()
    for frame in (images.letterbox_probe(shape), np.random.default_rng(5).integers(0, 256, shape, dtype=np.uint8)):
        np.testing.assert_array_equal(images.letterbox_image(frame, 448, 448),
                                      image_tools.resize_with_pad(frame, 448, 448))


def test_openpi_contract_round_trips_and_pins_pixels():
    import json
    contract = json.loads(json.dumps(images.openpi_image_contract(OPENPI_CAMERAS)))
    assert contract["fingerprint"] == {"probe": "xorhash-v1", "sha256": OPENPI_720P_FINGERPRINT}
    assert images.require_openpi_image_contract(contract, ["video_overhead", "video_right"]) == contract


@pytest.mark.parametrize("tamper", [
    lambda c: c["fingerprint"].update(sha256="0" * 64),
    lambda c: c.update(model_image_shape=[224, 224, 3]),
    lambda c: c.update(resize_owner="openpi"),
    lambda c: c["cameras"]["video_right"].update(source_shape=[800, 1280, 3]),
    lambda c: c.update(schema_version=1),
])
def test_openpi_contract_mismatch_fails_closed(tamper):
    contract = images.openpi_image_contract(OPENPI_CAMERAS)
    tamper(contract)
    with pytest.raises(ValueError, match="contract"):
        images.require_openpi_image_contract(contract, ["video_overhead", "video_right"])
    with pytest.raises(ValueError, match="cameras"):
        images.require_openpi_image_contract(images.openpi_image_contract(OPENPI_CAMERAS), ["video_right", "video_overhead"])


def test_openpi_request_images_are_letterboxed_rgb_under_request_keys():
    rng = np.random.default_rng(9)
    frames = {name: rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8) for name in OPENPI_CAMERAS}
    contract = images.openpi_image_contract(OPENPI_CAMERAS)
    request = images.openpi_request_images(frames, contract)
    assert list(request) == ["observation/image", "observation/wrist_image"]
    np.testing.assert_array_equal(request["observation/wrist_image"],
                                  images.letterbox_image(frames["video_right"][..., ::-1], 448, 448))
    with pytest.raises(ValueError, match="expected uint8"):
        images.openpi_request_images({**frames, "video_right": frames["video_right"][:700]}, contract)






@pytest.mark.parametrize("unsupported", [{"masking_strategy": "robot"}, {"image_history_len": 2}])
def test_server_rejects_unimplemented_preprocessing(unsupported):
    with pytest.raises(ValueError, match="masking|single-frame checkpoint"):
        serve_policy.LocalPolicyServer(
            policy=None, stats={}, config={"policy_class": "Flow", "policy_config": {}, **unsupported},
            ckpt_path="synthetic", host="localhost", port=0,
            device=torch.device("cpu"), use_fill_in=False)




