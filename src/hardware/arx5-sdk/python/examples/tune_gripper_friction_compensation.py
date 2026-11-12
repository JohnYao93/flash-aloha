import curses
import math
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(ROOT_DIR)
FRICTION_TAG = "arx5_friction_compensation"
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5


@dataclass
class FrictionTerms:
    enabled: bool = False
    transition_velocity: float = 0.02
    constant_term: float = 0.0
    coulomb_coef: float = 0.0
    viscous_coef: float = 0.0


FIELDS = (
    ("transition_velocity", "transition vel", 0.001, 0.0, 1.0),
    ("constant_term", "constant", 0.01, -1.5, 1.5),
    ("coulomb_coef", "coulomb", 0.01, -1.5, 1.5),
    ("viscous_coef", "viscous", 0.01, -5.0, 5.0),
)


def clamp(value, low, high):
    return min(max(value, low), high)


def target_label(target_index):
    if target_index == 0:
        return "gripper"
    return f"joint {target_index - 1}"


def current_terms(target_index, gripper_terms, joint_terms):
    if target_index == 0:
        return gripper_terms
    return joint_terms[target_index - 1]


def body_friction_enabled(joint_terms):
    return any(terms.enabled for terms in joint_terms)


def set_body_friction_enabled(joint_terms, enabled):
    for terms in joint_terms:
        terms.enabled = enabled


def apply_terms(controller, gripper_terms, joint_terms, friction_enabled):
    controller.set_gripper_friction_compensation(
        gripper_terms.enabled,
        gripper_terms.transition_velocity,
        gripper_terms.constant_term,
        gripper_terms.coulomb_coef,
        gripper_terms.viscous_coef,
    )

    joint_dof = len(joint_terms)
    transition_velocity = np.zeros(joint_dof, dtype=np.float64)
    constant_term = np.zeros(joint_dof, dtype=np.float64)
    coulomb_coef = np.zeros(joint_dof, dtype=np.float64)
    viscous_coef = np.zeros(joint_dof, dtype=np.float64)
    any_joint_enabled = body_friction_enabled(joint_terms)

    for idx, terms in enumerate(joint_terms):
        if not terms.enabled:
            continue
        transition_velocity[idx] = terms.transition_velocity
        constant_term[idx] = terms.constant_term
        coulomb_coef[idx] = terms.coulomb_coef
        viscous_coef[idx] = terms.viscous_coef

    controller.set_joint_friction_compensation(
        any_joint_enabled,
        transition_velocity,
        constant_term,
        coulomb_coef,
        viscous_coef,
    )
    controller.set_friction_compensation(friction_enabled)


def reset_terms_preserving_transition(terms):
    return FrictionTerms(transition_velocity=terms.transition_velocity)


def parse_bool_attribute(element, name):
    value = element.get(name)
    if value is None:
        raise ValueError(f"<{element.tag}> is missing '{name}'")
    normalized = value.strip().lower()
    if normalized not in ("true", "false", "1", "0"):
        raise ValueError(f"<{element.tag}> attribute '{name}' must be true or false")
    return normalized in ("true", "1")


def parse_float_attribute(element, name):
    value = element.get(name)
    if value is None:
        raise ValueError(f"<{element.tag}> is missing '{name}'")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"<{element.tag}> attribute '{name}' must be finite")
    return parsed


def terms_from_element(element):
    terms = FrictionTerms(
        enabled=parse_bool_attribute(element, "enabled"),
        transition_velocity=parse_float_attribute(element, "transition_velocity"),
        constant_term=parse_float_attribute(element, "constant_term"),
        coulomb_coef=parse_float_attribute(element, "coulomb_coef"),
        viscous_coef=parse_float_attribute(element, "viscous_coef"),
    )
    validate_enabled_terms(terms, element.tag)
    return terms


def validate_enabled_terms(terms, label):
    if terms.enabled and terms.transition_velocity <= 0.0:
        raise ValueError(f"enabled {label} requires transition_velocity > 0")


def validate_robot_root(root, model):
    if root.tag != "robot":
        raise ValueError("URDF is missing its <robot> root")
    if root.get("name") != model:
        raise ValueError(f"URDF robot is '{root.get('name')}', expected '{model}'")


def terms_attributes(terms):
    return {
        "enabled": str(terms.enabled).lower(),
        "transition_velocity": format(terms.transition_velocity, ".12g"),
        "constant_term": format(terms.constant_term, ".12g"),
        "coulomb_coef": format(terms.coulomb_coef, ".12g"),
        "viscous_coef": format(terms.viscous_coef, ".12g"),
    }


def build_friction_element(friction_enabled, gripper_terms, joint_terms):
    validate_enabled_terms(gripper_terms, "gripper")
    friction = ET.Element(FRICTION_TAG, {"enabled": str(friction_enabled).lower()})
    for index, terms in enumerate(joint_terms):
        validate_enabled_terms(terms, f"joint {index}")
        attributes = {"index": str(index)}
        attributes.update(terms_attributes(terms))
        ET.SubElement(friction, "joint", attributes)
    ET.SubElement(friction, "gripper", terms_attributes(gripper_terms))
    ET.indent(friction, space="    ")
    return friction


def replace_friction_element(path, friction):
    path_obj = Path(path)
    original = path_obj.read_text()
    serialized = ET.tostring(friction, encoding="unicode", short_empty_elements=True)
    serialized = "\n".join(f"    {line}" for line in serialized.splitlines())

    start = original.find(f"<{FRICTION_TAG}")
    if start >= 0:
        line_start = original.rfind("\n", 0, start) + 1
        closing_tag = f"</{FRICTION_TAG}>"
        end = original.find(closing_tag, start)
        if end < 0:
            raise ValueError(f"URDF has an unterminated <{FRICTION_TAG}> block")
        end += len(closing_tag)
        updated = original[:line_start] + serialized + original[end:]
    else:
        robot_end = original.rfind("</robot>")
        if robot_end < 0:
            raise ValueError("URDF is missing its </robot> closing tag")
        updated = original[:robot_end] + serialized + "\n" + original[robot_end:]

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path_obj.parent, prefix=f".{path_obj.name}.", delete=False) as temp:
            temp.write(updated)
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = Path(temp.name)
        os.chmod(temp_path, path_obj.stat().st_mode & 0o7777)
        os.replace(temp_path, path_obj)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def save_friction_config(path, model, friction_enabled, gripper_terms, joint_terms):
    root = ET.parse(path).getroot()
    validate_robot_root(root, model)
    if len(root.findall(FRICTION_TAG)) > 1:
        raise ValueError(f"URDF has multiple <{FRICTION_TAG}> blocks")
    replace_friction_element(path, build_friction_element(friction_enabled, gripper_terms, joint_terms))


def load_friction_config(path, model, joint_dof):
    root = ET.parse(path).getroot()
    validate_robot_root(root, model)
    friction_elements = root.findall(FRICTION_TAG)
    if not friction_elements:
        raise ValueError(f"URDF has no <{FRICTION_TAG}> block")
    if len(friction_elements) > 1:
        raise ValueError(f"URDF has multiple <{FRICTION_TAG}> blocks")
    friction = friction_elements[0]

    friction_enabled = parse_bool_attribute(friction, "enabled")
    joint_terms = [None] * joint_dof
    for joint in friction.findall("joint"):
        index = int(joint.get("index", "-1"))
        if index < 0 or index >= joint_dof:
            raise ValueError(f"joint friction index {index} is outside joint_dof {joint_dof}")
        if joint_terms[index] is not None:
            raise ValueError(f"duplicate joint friction index {index}")
        joint_terms[index] = terms_from_element(joint)
    missing = [str(index) for index, terms in enumerate(joint_terms) if terms is None]
    if missing:
        raise ValueError(f"missing joint friction indexes: {', '.join(missing)}")

    grippers = friction.findall("gripper")
    if not grippers:
        raise ValueError("URDF friction block has no <gripper>")
    if len(grippers) > 1:
        raise ValueError("URDF friction block has multiple <gripper> elements")
    return friction_enabled, terms_from_element(grippers[0]), joint_terms


def draw(
    stdscr,
    model,
    interface,
    urdf_path,
    last_config_status,
    target_index,
    gripper_terms,
    joint_terms,
    selected_field,
    step_scale,
    state,
    paused,
    gravity_enabled,
    friction_enabled,
    arm_kd_scale,
    command_gripper_torque,
    command_joint_torque,
):
    terms = current_terms(target_index, gripper_terms, joint_terms)
    enabled_joints = [str(idx) for idx, item in enumerate(joint_terms) if item.enabled]
    enabled_joints_text = ",".join(enabled_joints) if enabled_joints else "none"

    stdscr.erase()
    stdscr.addstr(0, 0, f"Passive friction backdrive tuner  model={model}  interface={interface}")
    stdscr.addstr(
        1,
        0,
        "q quit | [/ ] target | arrows tune | SPACE master friction | B body friction | G gripper friction | b/v torque bump +/- | g gravity | s save | l load | p pause | 0 zero field | r/R reset | +/- step",
    )
    stdscr.addstr(
        3,
        0,
        f"target: {target_label(target_index)}    target included: {'YES' if terms.enabled else 'NO '}    friction: {'ON ' if friction_enabled else 'OFF'}    gravity: {'ON ' if gravity_enabled else 'OFF'}    step scale: {step_scale:g}    paused: {'YES' if paused else 'NO'}",
    )
    stdscr.addstr(
        4,
        0,
        f"body friction: {'ON ' if body_friction_enabled(joint_terms) else 'OFF'} (enabled joints: {enabled_joints_text})    gripper friction: {'ON ' if gripper_terms.enabled else 'OFF'}",
    )
    stdscr.addstr(5, 0, f"URDF: {urdf_path}    {last_config_status}")

    for idx, (attr, label, base_step, low, high) in enumerate(FIELDS):
        prefix = ">" if idx == selected_field else " "
        value = getattr(terms, attr)
        stdscr.addstr(
            7 + idx,
            0,
            f"{prefix} {label:16s} {value:+.5f}   step {base_step * step_scale:g}   range [{low:g}, {high:g}]",
        )

    selected_joint_torque = 0.0 if target_index == 0 else command_joint_torque[target_index - 1]
    stdscr.addstr(13, 0, f"commanded bump torque: gripper {command_gripper_torque:+.5f} Nm   selected joint {selected_joint_torque:+.5f} Nm")
    stdscr.addstr(
        14,
        0,
        f"gripper pos: {state.gripper_pos:+.5f} m   vel: {state.gripper_vel:+.5f} m/s   torque: {state.gripper_torque:+.5f} Nm",
    )
    if target_index > 0:
        joint_idx = target_index - 1
        stdscr.addstr(
            15,
            0,
            f"selected joint {joint_idx}: pos {state.pos()[joint_idx]:+.5f} rad   vel {state.vel()[joint_idx]:+.5f} rad/s   torque {state.torque()[joint_idx]:+.5f} Nm",
        )
    else:
        stdscr.addstr(15, 0, "selected target is gripper")
    stdscr.addstr(17, 0, "Manually backdrive the selected target, or use b/v for brief torque bumps.")
    stdscr.addstr(18, 0, "If any joint pulls on its own or feels unstable, press SPACE to disable friction or R to reset all.")
    stdscr.refresh()


def run_tuner(
    stdscr,
    controller,
    model,
    interface,
    loop_dt,
    arm_kd_scale,
    urdf_path,
    gripper_bump_torque,
    joint_bump_torque,
    bump_time,
):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    robot_config = controller.get_robot_config()
    last_config_status = ""
    if os.path.exists(urdf_path):
        try:
            friction_enabled, gripper_terms, joint_terms = load_friction_config(
                urdf_path, model, robot_config.joint_dof
            )
            last_config_status = "loaded on startup"
        except Exception as exc:
            friction_enabled = False
            gripper_terms = FrictionTerms()
            joint_terms = [FrictionTerms() for _ in range(robot_config.joint_dof)]
            last_config_status = f"load failed: {exc}"
    else:
        friction_enabled = False
        gripper_terms = FrictionTerms()
        joint_terms = [FrictionTerms() for _ in range(robot_config.joint_dof)]
        last_config_status = "URDF not found"
    apply_terms(controller, gripper_terms, joint_terms, friction_enabled)

    target_index = 0
    selected_field = 0
    step_scale = 1.0
    paused = False
    gravity_enabled = False
    pulse_target_index = 0
    pulse_direction = 0.0
    pulse_until = 0.0

    while True:
        key = stdscr.getch()
        changed_terms = False

        while key != -1:
            terms = current_terms(target_index, gripper_terms, joint_terms)
            if key in (ord("q"), 27):
                return
            if key == ord("["):
                target_index = (target_index - 1) % (robot_config.joint_dof + 1)
            elif key == ord("]"):
                target_index = (target_index + 1) % (robot_config.joint_dof + 1)
            elif key == curses.KEY_LEFT:
                selected_field = (selected_field - 1) % len(FIELDS)
            elif key == curses.KEY_RIGHT:
                selected_field = (selected_field + 1) % len(FIELDS)
            elif key in (curses.KEY_UP, curses.KEY_DOWN):
                attr, _, base_step, low, high = FIELDS[selected_field]
                direction = 1.0 if key == curses.KEY_UP else -1.0
                setattr(terms, attr, clamp(getattr(terms, attr) + direction * base_step * step_scale, low, high))
                changed_terms = True
            elif key == ord(" "):
                friction_enabled = not friction_enabled
                if friction_enabled and not (gripper_terms.enabled or any(item.enabled for item in joint_terms)):
                    gripper_terms.enabled = True
                    set_body_friction_enabled(joint_terms, True)
                    changed_terms = True
                controller.set_friction_compensation(friction_enabled)
            elif key == ord("B"):
                set_body_friction_enabled(joint_terms, not body_friction_enabled(joint_terms))
                changed_terms = True
            elif key == ord("G"):
                gripper_terms.enabled = not gripper_terms.enabled
                changed_terms = True
            elif key == ord("b"):
                pulse_target_index = target_index
                pulse_direction = 1.0
                pulse_until = time.monotonic() + bump_time
            elif key == ord("v"):
                pulse_target_index = target_index
                pulse_direction = -1.0
                pulse_until = time.monotonic() + bump_time
            elif key == ord("0"):
                attr, _, _, _, _ = FIELDS[selected_field]
                setattr(terms, attr, 0.0)
                changed_terms = True
            elif key == ord("r"):
                if target_index == 0:
                    gripper_terms = reset_terms_preserving_transition(gripper_terms)
                else:
                    joint_idx = target_index - 1
                    joint_terms[joint_idx] = reset_terms_preserving_transition(joint_terms[joint_idx])
                changed_terms = True
            elif key == ord("R"):
                gripper_terms = reset_terms_preserving_transition(gripper_terms)
                joint_terms = [reset_terms_preserving_transition(item) for item in joint_terms]
                changed_terms = True
            elif key in (ord("+"), ord("=")):
                step_scale = min(step_scale * 10.0, 100.0)
            elif key in (ord("-"), ord("_")):
                step_scale = max(step_scale / 10.0, 0.01)
            elif key == ord("g"):
                gravity_enabled = not gravity_enabled
                controller.set_gravity_compensation(gravity_enabled)
            elif key == ord("s"):
                try:
                    save_friction_config(urdf_path, model, friction_enabled, gripper_terms, joint_terms)
                    last_config_status = "saved to URDF"
                except Exception as exc:
                    last_config_status = f"save failed: {exc}"
            elif key == ord("l"):
                try:
                    friction_enabled, gripper_terms, joint_terms = load_friction_config(
                        urdf_path, model, robot_config.joint_dof
                    )
                    changed_terms = True
                    last_config_status = "loaded from URDF"
                except Exception as exc:
                    last_config_status = f"load failed: {exc}"
            elif key == ord("p"):
                paused = not paused
            key = stdscr.getch()

        if changed_terms:
            apply_terms(controller, gripper_terms, joint_terms, friction_enabled)

        state = controller.get_joint_state()
        command_gripper_torque = 0.0
        command_joint_torque = np.zeros(robot_config.joint_dof, dtype=np.float64)

        if not paused:
            now = time.monotonic()
            if now < pulse_until:
                if pulse_target_index == 0:
                    command_gripper_torque = pulse_direction * gripper_bump_torque
                else:
                    command_joint_torque[pulse_target_index - 1] = pulse_direction * joint_bump_torque

            cmd = arx5.JointState(robot_config.joint_dof)
            cmd.torque()[:] = command_joint_torque
            cmd.gripper_torque = command_gripper_torque
            cmd.timestamp = controller.get_timestamp() + loop_dt
            controller.set_joint_cmd(cmd)

        draw(
            stdscr,
            model,
            interface,
            urdf_path,
            last_config_status,
            target_index,
            gripper_terms,
            joint_terms,
            selected_field,
            step_scale,
            state,
            paused,
            gravity_enabled,
            friction_enabled,
            arm_kd_scale,
            command_gripper_torque,
            command_joint_torque,
        )
        time.sleep(loop_dt)


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option("--gripper-kp", default=0.0, show_default=True, type=float, help="Position gain. Keep 0 for free backdriving.")
@click.option("--gripper-kd", default=0.0, show_default=True, type=float, help="Velocity gain. Keep 0 for free backdriving.")
@click.option("--loop-dt", default=0.02, show_default=True, type=float, help="Command loop period in seconds.")
@click.option("--arm-kd-scale", default=0.0, show_default=True, type=float, help="Scale on default joint damping. 0 matches the existing force-compensation examples.")
@click.option(
    "--urdf-path",
    "--config-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="URDF used for loading and saving friction terms. Defaults to the selected model's URDF.",
)
@click.option("--gripper-bump-torque", default=0.03, show_default=True, type=float, help="Torque pulse magnitude for gripper bump tests, in Nm.")
@click.option("--joint-bump-torque", default=0.2, show_default=True, type=float, help="Torque pulse magnitude for arm-joint bump tests, in Nm.")
@click.option("--bump-time", default=0.15, show_default=True, type=float, help="Duration of each b/v torque pulse in seconds.")
def main(
    model,
    interface,
    gripper_kp,
    gripper_kd,
    loop_dt,
    arm_kd_scale,
    urdf_path,
    gripper_bump_torque,
    joint_bump_torque,
    bump_time,
):
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    if urdf_path is None:
        urdf_path = robot_config.urdf_path
    if robot_config.gripper_motor_type != arx5.MotorType.DM_J4310:
        raise click.ClickException("This tuner only supports DM_J4310 gripper motors.")

    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.background_send_recv = True
    controller_config.gravity_compensation = False
    controller_config.friction_compensation = False
    controller_config.joint_friction_compensation = False
    controller_config.joint_friction_transition_velocity = np.zeros(robot_config.joint_dof, dtype=np.float64)
    controller_config.joint_friction_constant_term = np.zeros(robot_config.joint_dof, dtype=np.float64)
    controller_config.joint_friction_coulomb_coef = np.zeros(robot_config.joint_dof, dtype=np.float64)
    controller_config.joint_friction_viscous_coef = np.zeros(robot_config.joint_dof, dtype=np.float64)
    controller_config.gripper_friction_compensation = False
    controller_config.gripper_friction_transition_velocity = 0.02
    controller_config.gripper_friction_constant_term = 0.0
    controller_config.gripper_friction_coulomb_coef = 0.0
    controller_config.gripper_friction_viscous_coef = 0.0

    controller = arx5.Arx5JointController(robot_config, controller_config, interface)
    controller.set_log_level(arx5.LogLevel.WARNING)

    gain = arx5.Gain(robot_config.joint_dof)
    gain.kp()[:] = 0.0
    gain.kd()[:] = controller_config.default_kd * arm_kd_scale
    gain.gripper_kp = gripper_kp
    gain.gripper_kd = gripper_kd
    controller.set_gain(gain)

    try:
        curses.wrapper(
            run_tuner,
            controller,
            model,
            interface,
            loop_dt,
            arm_kd_scale,
            urdf_path,
            gripper_bump_torque,
            joint_bump_torque,
            bump_time,
        )
    finally:
        controller.set_friction_compensation(False)
        zero = np.zeros(robot_config.joint_dof, dtype=np.float64)
        controller.set_joint_friction_compensation(False, zero, zero, zero, zero)
        controller.set_gripper_friction_compensation(False, 0.0, 0.0, 0.0, 0.0)
        controller.set_to_damping()


if __name__ == "__main__":
    main()
