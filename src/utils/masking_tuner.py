"""Interactive adjustment of a masking strategy's projected polygons (no robot or camera code here).

``MaskTuner`` holds one strategy of the masking config, lets a caller move or scale the
corners of its ``project`` operations, renders the result over camera frames, and writes a
complete masking config with the tuned corners. ``scripts/setup/tune_masking.py`` drives it
from a recorded episode or from live cameras.
"""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml

from config.masking_config import load_masking_config, masking_config_path
from utils.masking import Masker

AXES = {'x': 0, 'y': 1, 'z': 2}
SELECTED, OTHER = (0, 255, 255), (255, 0, 255)  # BGR outline colours


class MaskTuner:
    def __init__(self, strategy, config_path=None, step=0.01):
        self.source = masking_config_path(config_path)
        self.masker = Masker(strategy, self.source)
        if not self.masker.is_projected:
            raise ValueError(f'Masking strategy {strategy!r} has no projected polygons to tune')
        self.strategy = strategy
        self.projects = [operation['project'] for operation in self.masker.operations if 'project' in operation]
        self._source_projects = deepcopy(self.projects)  # as written in the source file, to find them again
        self._loaded = deepcopy(self.projects)
        self.operation, self.corner, self.step = 0, None, float(step)  # corner None: the whole polygon

    # --- selection and edits ----------------------------------------------------------------

    @property
    def selected(self):
        return self.projects[self.operation]

    def select_operation(self, offset):
        self.operation = (self.operation + offset) % len(self.projects)
        self.corner = None

    def select_corner(self):
        """Cycle: whole polygon, then each corner in turn."""
        count = len(self.selected['corners'])
        self.corner = 0 if self.corner is None else (None if self.corner + 1 >= count else self.corner + 1)

    def move(self, axis, steps):
        """Shift the selected corner, or every corner, by ``steps`` * step metres along x, y or z."""
        targets = self.selected['corners'] if self.corner is None else [self.selected['corners'][self.corner]]
        for corner in targets:
            corner[AXES[axis]] = round(float(corner[AXES[axis]]) + steps * self.step, 6)

    def scale(self, factor):
        """Scale the selected polygon about its centroid."""
        corners = np.asarray(self.selected['corners'], dtype=float)
        centre = corners.mean(axis=0)
        for corner, value in zip(self.selected['corners'], centre + (corners - centre) * factor):
            corner[:] = [round(float(v), 6) for v in value]

    def scale_step(self, factor):
        self.step = float(np.clip(self.step * factor, 0.001, 0.2))

    def reset(self):
        for project, loaded in zip(self.projects, self._loaded):
            project['corners'][:] = deepcopy(loaded['corners'])

    @property
    def changed(self):
        return any(project['corners'] != loaded['corners'] for project, loaded in zip(self.projects, self._loaded))

    # --- display ----------------------------------------------------------------------------

    def describe(self):
        project = self.selected
        target = 'all corners' if self.corner is None else f'corner {self.corner} {project["corners"][self.corner]}'
        return (f'{self.strategy}  op {self.operation + 1}/{len(self.projects)}: {project["camera"]} camera, '
                f'{project["frame"]} of {project["arm"]} arm  |  {target}  |  step {self.step * 100:.1f} cm'
                + ('  |  unsaved changes' if self.changed else ''))

    def render(self, images, qpos, scale=2):
        """One BGR image: every camera with its mask dimmed, polygon outlines, and the selection highlighted."""
        masked, _ = self.masker.mask(images, qpos)
        tiles = []
        for name, image in images.items():
            size = image.shape[1::-1]
            view = cv2.addWeighted(np.ascontiguousarray(image), 0.35, np.ascontiguousarray(masked[name]), 0.65, 0)
            view = cv2.resize(view, (size[0] * scale, size[1] * scale), interpolation=cv2.INTER_NEAREST)
            for index, project in enumerate(self.projects):
                active = index == self.operation
                for points in self.masker._project_operation(project, qpos, size).get(name, []):
                    points = np.clip(points, -10 * size[0], 10 * size[0]) * scale
                    cv2.polylines(view, [points.reshape(-1, 1, 2)], True, SELECTED if active else OTHER, 2 if active else 1)
                    if active and self.corner is not None:
                        x, y = points[self.corner]
                        cv2.drawMarker(view, (int(x), int(y)), SELECTED, cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(view, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(view)
        height = max(tile.shape[0] for tile in tiles)
        tiles = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT) for tile in tiles]
        canvas = np.hstack(tiles)
        banner = np.zeros((24, canvas.shape[1], 3), np.uint8)
        cv2.putText(banner, self.describe(), (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([banner, canvas])

    # --- saving -----------------------------------------------------------------------------

    def save(self, output=None):
        """Write a complete masking config with the tuned corners; returns its path.

        Every strategy that used one of the tuned operations gets the new corners, so variants of
        a strategy stay identical. The default output sits beside the source config
        (``<name>.tuned.yaml``): select it with FLASH_ALOHA_MASKING_CONFIG, or replace the source.
        A checkpoint trained with the previous corners is refused with the new ones.
        """
        output = Path(output).expanduser().resolve() if output else self.source.with_suffix('.tuned.yaml')
        document = yaml.safe_load(self.source.read_text())
        document.pop('definitions', None)  # anchors are expanded; strategies carry their own operations
        for operations in document['strategies'].values():
            for operation in operations:
                for original, tuned in zip(self._source_projects, self.projects):
                    if operation.get('project') == original:
                        operation['project']['corners'] = deepcopy(tuned['corners'])
        if output.parent != self.source.parent and 'calibration' in document:
            resolved = load_masking_config(self.source)['calibration']  # absolute calibration file paths
            calibration = document['calibration']
            calibration['urdf'] = resolved['urdf']
            calibration['overhead_camera']['intrinsics'] = resolved['overhead_camera']['intrinsics']
            for name, arm in calibration['arms'].items():
                for key in ('intrinsics', 'camera_on_end_effector', 'base_in_overhead'):
                    arm[key] = resolved['arms'][name][key]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f'# Tuned from {self.source.name} with scripts/setup/tune_masking.py (strategy {self.strategy}).\n'
                          + yaml.safe_dump(document, sort_keys=False))
        load_masking_config(output)  # the written file must be a valid masking config
        self._loaded = deepcopy(self.projects)
        return output
