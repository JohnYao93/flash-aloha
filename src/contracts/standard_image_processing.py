"""Standard deterministic image processing for training and deployment.

Storage decoders supply BGR; LeRobot and OpenPI requests can supply RGB.
Spatial processing happens before channel conversion and float32 scaling.
Random augmentations are deliberately not part of this module.
Custom profiles must compose these operations, not copy or replace the pipeline.
"""
from dataclasses import dataclass
import hashlib

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageSpec:
    width: int | None = None
    height: int | None = None
    color: str = "RGB"
    grayscale: bool = False
    interpolation: int = cv2.INTER_AREA
    # Explicit source coordinates, applied before resize: top, left, height, width.
    crop: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        if (self.width is None) != (self.height is None):
            raise ValueError("Image width and height must be specified together")
        if self.width is not None and (self.width <= 0 or self.height <= 0):
            raise ValueError("Image dimensions must be positive")
        if self.color not in ("RGB", "BGR"):
            raise ValueError(f"Unsupported output color space: {self.color}")


def policy_image_spec(config):
    """Resolve the standard Zarr ACT/Flow profile from checkpoint configuration."""
    policy = config.get("policy_config", {})
    task = config.get("task_config", {})
    for dimension in ("width", "height"):
        if (config.get(dimension) is not None and policy.get(dimension) is not None
                and config[dimension] != policy[dimension]):
            raise ValueError(f"Checkpoint {dimension} conflicts with policy_config: "
                             f"{config[dimension]} != {policy[dimension]}")
    return ImageSpec(
        width=policy.get("width", config.get("width")),
        height=policy.get("height", config.get("height")),
        color="BGR" if config.get("aloha", task.get("aloha", False)) else "RGB",
        grayscale=bool(config.get("grayscale", False)),
    )


def masking_strategy(config, requested=None):
    """A deployment CLI may confirm a saved mask, but cannot change it."""
    saved = config.get("task_config", {}).get("masking_strategy", config.get("masking_strategy"))
    if requested is not None and requested != saved:
        raise ValueError(f"Masking override {requested!r} differs from training {saved!r}")
    return saved



def convert_color(image, source_color, target_color):
    if source_color not in ("RGB", "BGR") or target_color not in ("RGB", "BGR"):
        raise ValueError(f"Unsupported color conversion {source_color!r} -> {target_color!r}")
    return np.ascontiguousarray(image if source_color == target_color else image[..., ::-1])


def resize_image(image, width, height, *, interpolation=cv2.INTER_AREA):
    if width is None and height is None:
        return image
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("Image width and height must both be positive")
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (int(width), int(height)), interpolation=interpolation)


def prepare_image(image, spec=ImageSpec(), *, source_color="BGR"):
    """Return uint8 HWC in the profile's channel order; never mutate the input."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HxWx3 image; got {image.shape}, {image.dtype}")
    if spec.crop is not None:
        top, left, height, width = spec.crop
        if min(top, left) < 0 or min(height, width) <= 0 or top + height > image.shape[0] or left + width > image.shape[1]:
            raise ValueError(f"Crop {spec.crop} is outside image {image.shape}")
        image = image[top:top + height, left:left + width]
    image = resize_image(image, spec.width, spec.height, interpolation=spec.interpolation)
    return convert_color(image, source_color, spec.color)


def image_to_chw(image, spec=ImageSpec(), *, source_color="BGR"):
    image = prepare_image(image, spec, source_color=source_color)
    result = image.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
    if spec.grayscale:
        result = np.repeat(result.mean(axis=0, keepdims=True), 3, axis=0)
    return np.ascontiguousarray(result)


def inference_images(image_dict, *, camera_names=None, is_double_arm=False,
                     aloha=False, grayscale=False, normalize_resnet=False,
                     image_history_buffer=None, device="cuda"):
    """Shared native inference adapter. Inputs have already been resized/masked."""
    import torch
    if normalize_resnet:
        raise ValueError("Backbone normalization belongs in model.forward, not preprocessing")
    names = list(camera_names) if camera_names is not None else ["video_left", "video_overhead"] + (["video_right"] if is_double_arm else [])
    if image_history_buffer is not None:
        raise ValueError('Multi-frame image history is no longer supported')
    frames = {name: [image_dict[name]] for name in names}
    spec = ImageSpec(color="BGR" if aloha else "RGB", grayscale=grayscale)
    arrays = [image_to_chw(frame, spec) for name in names for frame in frames[name]]
    return torch.from_numpy(np.stack(arrays)).unsqueeze(0).to(device)


def letterbox_image(image, width, height, *, interpolation=cv2.INTER_LINEAR):
    """Aspect-preserving resize centered on a zero canvas.

    The geometry, including its float truncation, is that of
    openpi_client.image_tools.resize_with_pad, which OpenPI training applies to
    native frames. The contract fingerprint pins the resulting pixels.
    """
    if image.shape[:2] == (height, width):
        return image
    source_height, source_width = image.shape[:2]
    ratio = max(source_width / width, source_height / height)
    content_height, content_width = int(source_height / ratio), int(source_width / ratio)
    content = resize_image(image, content_width, content_height, interpolation=interpolation)
    canvas = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    top, left = max(0, int((height - content_height) / 2)), max(0, int((width - content_width) / 2))
    canvas[top:top + content_height, left:left + content_width] = content
    return canvas


# OpenPI profile: the robot client letterboxes each native camera frame to the
# model resolution and sends RGB uint8; the server checks the contract and never resizes.
OPENPI_PROFILE = "openpi_letterbox_448"
OPENPI_IMAGE_SIZE = 448
OPENPI_CONTRACT_SCHEMA = 2
OPENPI_FINGERPRINT_PROBE = "xorhash-v1"


def letterbox_probe(shape):
    """Deterministic high-frequency uint8 image from integer arithmetic only.

    Other implementations regenerate it exactly (OpenPI's trainer recomputes the
    fingerprint with its own resize and OpenCV build before training).
    """
    height, width, channels = shape
    y, x, c = np.ogrid[:height, :width, :channels]
    value = (x * 73856093) ^ (y * 19349663) ^ (c * 83492791)
    return ((value ^ (value >> 13)) & 255).astype(np.uint8)


def letterbox_fingerprint(source_shapes, size=OPENPI_IMAGE_SIZE):
    digest = hashlib.sha256()
    for shape in sorted({tuple(int(n) for n in shape) for shape in source_shapes}):
        digest.update(np.ascontiguousarray(letterbox_image(letterbox_probe(shape), size, size)).tobytes())
    return digest.hexdigest()


def openpi_image_contract(cameras):
    """cameras: {camera name: {request_key, source_shape [h, w, 3]}} in request order."""
    cameras = {name: {"request_key": str(camera["request_key"]),
                      "source_shape": [int(n) for n in camera["source_shape"]]}
               for name, camera in cameras.items()}
    if not cameras or any(len(c["source_shape"]) != 3 or c["source_shape"][2] != 3 for c in cameras.values()):
        raise ValueError(f"OpenPI cameras need HxWx3 source shapes; got {cameras}")
    if len({c["request_key"] for c in cameras.values()}) != len(cameras):
        raise ValueError("OpenPI cameras must use distinct request keys")
    return {
        "schema_version": OPENPI_CONTRACT_SCHEMA,
        "profile": OPENPI_PROFILE,
        "resize_owner": "client",
        "color_order": "RGB",
        "wire_dtype": "uint8",
        "layout": "HWC",
        "model_image_shape": [OPENPI_IMAGE_SIZE, OPENPI_IMAGE_SIZE, 3],
        "resize": {"method": "letterbox", "interpolation": "INTER_LINEAR", "pad_value": 0, "anchor": "center",
                   "reference": "openpi_client.image_tools.resize_with_pad"},
        "cameras": cameras,
        "fingerprint": {"probe": OPENPI_FINGERPRINT_PROBE,
                        "sha256": letterbox_fingerprint(c["source_shape"] for c in cameras.values())},
    }


def require_openpi_image_contract(advertised, camera_names):
    """Rebuild the contract here (fingerprint with this process's OpenCV) and require equality."""
    if not isinstance(advertised, dict) or advertised.get("schema_version") != OPENPI_CONTRACT_SCHEMA:
        raise ValueError("OpenPI server does not advertise the client-side image preprocessing contract "
                         f"(schema {OPENPI_CONTRACT_SCHEMA}); retrain/export with it instead of sending unchecked pixels")
    if list(advertised.get("cameras", {})) != list(camera_names):
        raise ValueError(f"OpenPI contract cameras {list(advertised.get('cameras', {}))} differ from task cameras "
                         f"{list(camera_names)}")
    expected = openpi_image_contract(advertised["cameras"])
    if advertised != expected:
        differing = sorted(key for key in set(advertised) | set(expected) if advertised.get(key) != expected.get(key))
        raise ValueError(f"OpenPI image preprocessing contract mismatch in {differing}; "
                         "this client would not reproduce the checkpoint's training pixels")
    return expected


def openpi_request_images(images, contract):
    """Native BGR uint8 camera frames -> {request_key: model-resolution RGB uint8 letterbox}."""
    size = contract["model_image_shape"][0]
    request = {}
    for name, camera in contract["cameras"].items():
        frame = np.asarray(images[name])
        if list(frame.shape) != camera["source_shape"] or frame.dtype != np.uint8:
            raise ValueError(f"{name}: expected uint8 {camera['source_shape']}; got {frame.dtype} {list(frame.shape)}")
        request[camera["request_key"]] = letterbox_image(prepare_image(frame, source_color="BGR"), size, size)
    return request






def require_single_frame(*configs):
    """Reject historical-image checkpoints instead of silently changing their inputs."""
    for config in configs:
        if not isinstance(config, dict):
            continue
        for candidate in (config, config.get('policy_config', {}), config.get('task_config', {})):
            if candidate.get('image_history_len') not in (None, 0):
                raise ValueError('Multi-frame image history is no longer supported; use a single-frame checkpoint')
            if any('_t-' in str(name) for key in ('camera_names', 'camera_names_for_model')
                   for name in candidate.get(key, [])):
                raise ValueError('Historical virtual cameras are no longer supported')
