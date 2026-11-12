#!/usr/bin/env python3
"""Optional: adjust a masking strategy's polygons while watching the masks.

    # On a recorded episode (no robot or camera needed):
    python scripts/setup/tune_masking.py end_effector --episode data/TASK/episode_00000.zarr
    # Live, operator-run (ARX): camera devices and hand-guided arms
    python scripts/setup/tune_masking.py end_effector --live

Each camera is shown at the policy resolution with the mask dimmed and its polygons
outlined; the selected polygon is highlighted. Keys:

    [ ]      previous / next projected polygon of the strategy
    c        select the whole polygon, then each corner in turn
    x X      move the selection along -x / +x (metres, in the polygon's frame); y Y and z Z likewise
    > <      grow / shrink the selected polygon about its centre
    + -      double / halve the step
    r        discard unsaved changes
    s        save a tuned masking config (see --output)
    , .      previous / next frame (episode);  space  play / pause (episode)
    q        quit

Saving writes a complete masking config next to the source one (<name>.tuned.yaml by
default). Use it with FLASH_ALOHA_MASKING_CONFIG=<file>, or replace the source config.
Checkpoints trained with the previous polygons are refused with the new ones: retrain.

In --live mode each arm is homed, then put in gravity-compensated passive mode so you can
guide it by hand; on exit it returns home. Ctrl+C or any failure enters damping until 'h'
homes the arms (docs/safe_exit.md). Stop the camera servers first.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from config.masking_config import load_masking_config  # noqa: E402
from utils.masking_tuner import MaskTuner  # noqa: E402

WINDOW = 'Masking tuner'
DEFAULT_CAN_INTERFACES = {'left': 'can0', 'right': 'can1'}


class EpisodeSource:
    """Frames and qpos of a recorded episode, read and resized as training reads them."""

    def __init__(self, path, cameras, size):
        from data_processing.episode_data import open_episode
        from policy.training_utils import prepare_policy_frame
        self._prepare, self.size = prepare_policy_frame, size
        self._context = open_episode(str(path))
        self.root = self._context.__enter__()
        self.compressed = self.root.attrs.get('compress', False)
        self.qpos = np.asarray(self.root['/observations/qpos'][()])
        self.cameras = [name for name in cameras if f'observations/images/{name}' in self.root]
        if not self.cameras:
            raise ValueError(f'{path} has none of the masking config cameras {list(cameras)}')
        self.index, self.playing = 0, False

    def __len__(self):
        return len(self.qpos)

    def step(self, offset):
        self.index = int(np.clip(self.index + offset, 0, len(self) - 1))

    def read(self):
        if self.playing:
            self.step(1)
        images = {name: self._prepare(self.root[f'/observations/images/{name}'][self.index], self.compressed, *self.size)
                  for name in self.cameras}
        return images, self.qpos[self.index], f'frame {self.index + 1}/{len(self)}'

    def close(self):
        self._context.__exit__(None, None, None)


class LiveSource:
    """Camera devices and the joint positions of hand-guided ARX arms."""

    def __init__(self, controllers, cameras, size):
        self.controllers, self.size, self.captures = controllers, size, {}
        for name in cameras:
            capture = cv2.VideoCapture(f'/dev/{name}')
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open /dev/{name}; stop the camera servers first')
            self.captures[name] = capture

    def read(self):
        from contracts.standard_image_processing import resize_image
        images = {}
        for name, capture in self.captures.items():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Failed to read a frame from the {name} camera')
            images[name] = resize_image(frame, *self.size)
        qpos = []
        for controller in self.controllers:  # each arm: its joints, then its gripper
            state = controller.get_joint_state()
            qpos += [*np.asarray(state.pos()), float(state.gripper_pos)]
        return images, np.asarray(qpos, dtype=np.float32), 'live'

    def close(self):
        for capture in self.captures.values():
            capture.release()


def handle_key(key, tuner, source, output):
    """Apply one key press; returns False to quit."""
    character = chr(key) if 0 <= key < 256 else ''
    if not character:
        return True
    if character == 'q':
        return False
    if character in '[]':
        tuner.select_operation(-1 if character == '[' else 1)
    elif character == 'c':
        tuner.select_corner()
    elif character.lower() in ('x', 'y', 'z'):
        tuner.move(character.lower(), 1 if character.isupper() else -1)
    elif character in '<>':
        tuner.scale(1.05 if character == '>' else 1 / 1.05)
    elif character in '+=-_':
        tuner.scale_step(2 if character in '+=' else 0.5)
    elif character == 'r':
        tuner.reset()
    elif character == 's':
        print(f'Saved {tuner.save(output)}')
    elif isinstance(source, EpisodeSource):
        if character in ',.':
            source.playing = False
            source.step(-1 if character == ',' else 1)
        elif character == ' ':
            source.playing = not source.playing
    return True


def run(tuner, source, output, scale):
    print(__doc__.split('Keys:')[1].split('Saving writes')[0])
    while True:
        images, qpos, status = source.read()
        view = tuner.render(images, qpos, scale=scale)
        cv2.putText(view, status, (view.shape[1] - 150, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30)
        if key != -1 and not handle_key(key & 0xFF, tuner, source, output):
            break
    if tuner.changed:
        print('Unsaved changes were discarded.')


def run_live(args, tuner, setup, size):
    """Home, passive gravity compensation, tune, home; every failure goes through the safe exit."""
    from calibrate_arx_arm import build_controller, load_arx5_interface, safe_home, set_passive_mode
    from hardware.safe_exit import ARXDamping, damping_exit
    interfaces = args.can_interface or [DEFAULT_CAN_INTERFACES[name] for name in setup['arms']]
    if len(interfaces) != len(setup['arms']):
        raise SystemExit(f'--can-interface needs one interface per arm {list(setup["arms"])}')
    arx5 = load_arx5_interface()
    controllers, source, failed = [], None, False
    print("Ctrl+C or any failure: damping, then 'h' returns the arms home.")
    try:
        with damping_exit(lambda: ARXDamping(controllers) if controllers else None):
            for interface in interfaces:
                controllers.append(build_controller(arx5, args.model, interface, args.controller_fps)[2])
            for controller in controllers:
                controller.reset_to_home()
                set_passive_mode(controller)
            source = LiveSource(controllers, tuner.masker.cameras, size)
            run(tuner, source, args.output, args.scale)
    except KeyboardInterrupt:
        failed = True
        print('\nInterrupted.')
    except BaseException:
        failed = True
        raise
    finally:
        if source is not None:
            source.close()
        cv2.destroyAllWindows()
        if not failed:  # The failure exit already homed (or disabled in place).
            for controller in controllers:
                safe_home(controller)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('strategy', help='Masking strategy to tune (it must project at least one polygon)')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--episode', type=Path, help='Recorded episode to scrub through (no robot)')
    source.add_argument('--live', action='store_true', help='Live cameras and hand-guided ARX arms (operator-run)')
    parser.add_argument('--masking-config', help='Masking config YAML (default: calibration/masking/masking.yaml)')
    parser.add_argument('--output', type=Path, help='Tuned config to write (default: <config>.tuned.yaml beside it)')
    parser.add_argument('--width', type=int, help='Policy image width (default: the calibrated image size)')
    parser.add_argument('--height', type=int, help='Policy image height (default: the calibrated image size)')
    parser.add_argument('--scale', type=int, default=3, help='Display magnification')
    parser.add_argument('--step', type=float, default=0.01, help='Initial adjustment step in metres')
    parser.add_argument('--can-interface', nargs='+', help='Live: one CAN interface per arm, in the config\'s arm order')
    parser.add_argument('--model', default='L5', help='Live: ARX SDK model name')
    parser.add_argument('--controller-fps', type=float, default=500.0, help='Live: ARX controller rate')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.width is None) != (args.height is None):
        raise SystemExit('--width and --height go together')
    tuner = MaskTuner(args.strategy, args.masking_config, step=args.step)
    setup = load_masking_config(args.masking_config)['calibration']
    size = (args.width, args.height) if args.width else tuple(setup['image_size'])
    if args.live:
        run_live(args, tuner, setup, size)
        return 0
    source = EpisodeSource(args.episode, tuner.masker.cameras, size)
    try:
        run(tuner, source, args.output, args.scale)
    finally:
        source.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
