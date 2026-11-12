import curses
import ctypes
import math
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5


def clamp(value, low, high):
    return min(max(value, low), high)


def validate_robot_root(root, model):
    if root.tag != "robot":
        raise ValueError("URDF is missing its <robot> root")
    if root.get("name") != model:
        raise ValueError(f"URDF robot is '{root.get('name')}', expected '{model}'")


def load_link_masses(path, model):
    root = ET.parse(path).getroot()
    validate_robot_root(root, model)

    masses = OrderedDict()
    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            raise ValueError("URDF contains a <link> without a name")
        inertial = link.find("inertial")
        mass_element = None if inertial is None else inertial.find("mass")
        if mass_element is None or mass_element.get("value") is None:
            continue
        mass = float(mass_element.get("value"))
        if not math.isfinite(mass) or mass < 0.0:
            raise ValueError(f"link '{name}' mass must be finite and nonnegative")
        if name in masses:
            raise ValueError(f"duplicate link name '{name}'")
        masses[name] = mass

    if not masses:
        raise ValueError("URDF has no <link><inertial><mass> values to tune")
    return masses


def inverse_dynamics_link_names(path, model, base_link_name, eef_link_name):
    root = ET.parse(path).getroot()
    validate_robot_root(root, model)
    masses = load_link_masses(path, model)

    joints_by_child = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError("URDF contains a joint without parent and child links")
        child_name = child.get("link")
        if child_name in joints_by_child:
            raise ValueError(f"link '{child_name}' has multiple parent joints")
        joints_by_child[child_name] = (parent.get("link"), joint.get("type"))

    chain = []
    child_name = eef_link_name
    visited = set()
    while child_name != base_link_name:
        if child_name in visited:
            raise ValueError("URDF kinematic chain contains a cycle")
        visited.add(child_name)
        if child_name not in joints_by_child:
            raise ValueError(
                f"cannot trace a joint chain from '{eef_link_name}' to '{base_link_name}'"
            )
        parent_name, joint_type = joints_by_child[child_name]
        chain.append((child_name, joint_type))
        child_name = parent_name
    chain.reverse()

    active_indexes = [index for index, (_, joint_type) in enumerate(chain) if joint_type != "fixed"]
    if not active_indexes:
        raise ValueError("inverse-dynamics chain has no active joints")
    last_active_index = active_indexes[-1]
    names = [name for name, _ in chain[: last_active_index + 1] if name in masses]
    if not names:
        raise ValueError("inverse-dynamics chain has no tunable link masses")
    return names


def _mass_value_span(text, link_name):
    quoted_name = re.escape(link_name)
    link_pattern = re.compile(
        rf"<link\b(?=[^>]*\bname\s*=\s*(['\"]){quoted_name}\1)[^>]*>.*?</link\s*>",
        re.DOTALL,
    )
    links = list(link_pattern.finditer(text))
    if len(links) != 1:
        raise ValueError(f"expected exactly one <link name=\"{link_name}\"> block, found {len(links)}")

    link_match = links[0]
    inertial_match = re.search(r"<inertial\b[^>]*>.*?</inertial\s*>", link_match.group(0), re.DOTALL)
    if inertial_match is None:
        raise ValueError(f"link '{link_name}' has no <inertial> block")
    mass_match = re.search(r"<mass\b[^>]*/?>", inertial_match.group(0), re.DOTALL)
    if mass_match is None:
        raise ValueError(f"link '{link_name}' has no <mass> element")
    value_match = re.search(r"\bvalue\s*=\s*(['\"])([^'\"]+)\1", mass_match.group(0))
    if value_match is None:
        raise ValueError(f"link '{link_name}' mass has no value attribute")

    value_start = (
        link_match.start()
        + inertial_match.start()
        + mass_match.start()
        + value_match.start(2)
    )
    return value_start, value_start + len(value_match.group(2))


def render_link_masses(path, masses):
    original = Path(path).read_text()
    replacements = []
    for link_name, mass in masses.items():
        if not math.isfinite(mass) or mass < 0.0:
            raise ValueError(f"link '{link_name}' mass must be finite and nonnegative")
        start, end = _mass_value_span(original, link_name)
        replacements.append((start, end, format(mass, ".12g")))

    updated = original
    for start, end, value in sorted(replacements, reverse=True):
        updated = updated[:start] + value + updated[end:]
    return updated


def atomic_write(path, text):
    path_obj = Path(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path_obj.parent, prefix=f".{path_obj.name}.", delete=False
        ) as temp:
            temp.write(text)
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = Path(temp.name)
        os.chmod(temp_path, path_obj.stat().st_mode & 0o7777)
        os.replace(temp_path, path_obj)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def save_link_masses(path, model, masses):
    root = ET.parse(path).getroot()
    validate_robot_root(root, model)
    updated = render_link_masses(path, masses)

    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as check_file:
        check_file.write(updated)
        check_path = Path(check_file.name)
    try:
        check_root = ET.parse(check_path).getroot()
        validate_robot_root(check_root, model)
    finally:
        check_path.unlink()

    atomic_write(path, updated)


def write_solver_urdf(source_path, destination_path, masses):
    Path(destination_path).write_text(render_link_masses(source_path, masses))


def build_solver(robot_config, urdf_path):
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            return arx5.Arx5Solver(
                str(urdf_path),
                robot_config.joint_dof,
                robot_config.joint_pos_min,
                robot_config.joint_pos_max,
                robot_config.base_link_name,
                robot_config.eef_link_name,
                robot_config.gravity_vector,
            )
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def rebuild_solver(robot_config, source_path, solver_path, masses):
    write_solver_urdf(source_path, solver_path, masses)
    return build_solver(robot_config, solver_path)


def safe_addstr(stdscr, row, column, text):
    height, width = stdscr.getmaxyx()
    if row < 0 or row >= height or column < 0 or column >= width:
        return
    available = width - column
    if row == height - 1:
        available -= 1
    if available <= 0:
        return
    try:
        stdscr.addstr(row, column, str(text)[:available])
    except curses.error:
        # A terminal resize can occur between getmaxyx() and addstr().
        pass


def draw(
    stdscr,
    model,
    interface,
    urdf_path,
    status,
    link_names,
    masses,
    loaded_masses,
    target_index,
    mass_step,
    step_scale,
    gravity_enabled,
    friction_enabled,
    paused,
    state,
    gravity_torque,
):
    link_name = link_names[target_index]
    mass = masses[link_name]
    loaded_mass = loaded_masses[link_name]

    stdscr.erase()
    safe_addstr(stdscr, 0, 0, f"Gravity compensation mass tuner  model={model}  interface={interface}")
    safe_addstr(
        stdscr,
        1,
        0,
        "q quit | [/ ] link | up/down mass | SPACE gravity | s save | l load | p pause | r reset link | R reset all | +/- step",
    )
    safe_addstr(stdscr, 3, 0, f"target: {link_name} ({target_index + 1}/{len(link_names)})")
    safe_addstr(
        stdscr,
        4,
        0,
        f"mass: {mass:.6f} kg   loaded: {loaded_mass:.6f} kg   delta: {mass - loaded_mass:+.6f} kg   step: {mass_step * step_scale:g} kg",
    )
    safe_addstr(
        stdscr,
        5,
        0,
        f"gravity: {'ON ' if gravity_enabled else 'OFF'}   friction: {'ON ' if friction_enabled else 'OFF'}   damping: OFF   paused: {'YES' if paused else 'NO '}   {status}",
    )
    safe_addstr(stdscr, 6, 0, f"URDF: {urdf_path}")

    first_row = 8
    height, _ = stdscr.getmaxyx()
    visible_rows = max(0, min(len(link_names), height - first_row - 7))
    start = clamp(target_index - visible_rows // 2, 0, max(0, len(link_names) - visible_rows))
    for row, index in enumerate(range(start, start + visible_rows)):
        name = link_names[index]
        prefix = ">" if index == target_index else " "
        safe_addstr(
            stdscr,
            first_row + row,
            0,
            f"{prefix} {name:20s} {masses[name]:10.6f} kg   ({masses[name] - loaded_masses[name]:+.6f})",
        )

    telemetry_row = first_row + visible_rows + 1
    safe_addstr(
        stdscr,
        telemetry_row,
        0,
        "joint position: " + np.array2string(state.pos(), precision=3, suppress_small=True),
    )
    safe_addstr(
        stdscr,
        telemetry_row + 1,
        0,
        "joint velocity: " + np.array2string(state.vel(), precision=3, suppress_small=True),
    )
    safe_addstr(
        stdscr,
        telemetry_row + 2,
        0,
        "gravity torque: " + np.array2string(gravity_torque, precision=3, suppress_small=True),
    )
    safe_addstr(
        stdscr,
        telemetry_row + 4,
        0,
        "Move the arm through several poses; tune distal links before proximal links.",
    )
    safe_addstr(
        stdscr,
        telemetry_row + 5,
        0,
        "If the arm pulls away or oscillates, press SPACE immediately to disable gravity torque.",
    )
    stdscr.refresh()


def run_tuner(
    stdscr,
    controller,
    model,
    interface,
    loop_dt,
    urdf_path,
    mass_step,
    min_mass,
    max_mass,
):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    robot_config = controller.get_robot_config()
    all_masses = load_link_masses(urdf_path, model)
    link_names = inverse_dynamics_link_names(
        urdf_path,
        model,
        robot_config.base_link_name,
        robot_config.eef_link_name,
    )
    masses = OrderedDict((name, all_masses[name]) for name in link_names)
    loaded_masses = masses.copy()
    target_index = 0
    step_scale = 1.0
    gravity_enabled = False
    controller_config = controller.get_controller_config()
    friction_enabled = (
        controller_config.friction_compensation
        and controller_config.joint_friction_compensation
    )
    paused = False
    status = "loaded from URDF"

    with tempfile.TemporaryDirectory(prefix="arx5_gravity_tuner_") as temp_dir:
        solver_path = Path(temp_dir) / Path(urdf_path).name
        solver = rebuild_solver(robot_config, urdf_path, solver_path, masses)

        while True:
            key = stdscr.getch()

            while key != -1:
                link_name = link_names[target_index]
                if key in (ord("q"), 27):
                    return
                if key == ord("["):
                    target_index = (target_index - 1) % len(link_names)
                elif key == ord("]"):
                    target_index = (target_index + 1) % len(link_names)
                elif key in (curses.KEY_UP, curses.KEY_DOWN):
                    direction = 1.0 if key == curses.KEY_UP else -1.0
                    candidate_masses = masses.copy()
                    candidate_masses[link_name] = clamp(
                        candidate_masses[link_name] + direction * mass_step * step_scale,
                        min_mass,
                        max_mass,
                    )
                    try:
                        solver = rebuild_solver(
                            robot_config, urdf_path, solver_path, candidate_masses
                        )
                        masses = candidate_masses
                        status = "live gravity model updated"
                    except Exception as exc:
                        status = f"model update rejected: {exc}"
                elif key == ord(" "):
                    gravity_enabled = not gravity_enabled
                    status = "gravity enabled" if gravity_enabled else "gravity disabled"
                elif key in (ord("+"), ord("=")):
                    step_scale = min(step_scale * 10.0, 100.0)
                elif key in (ord("-"), ord("_")):
                    step_scale = max(step_scale / 10.0, 0.01)
                elif key == ord("r"):
                    candidate_masses = masses.copy()
                    candidate_masses[link_name] = loaded_masses[link_name]
                    try:
                        solver = rebuild_solver(
                            robot_config, urdf_path, solver_path, candidate_masses
                        )
                        masses = candidate_masses
                        status = f"reset {link_name}"
                    except Exception as exc:
                        status = f"model reset rejected: {exc}"
                elif key == ord("R"):
                    candidate_masses = loaded_masses.copy()
                    try:
                        solver = rebuild_solver(
                            robot_config, urdf_path, solver_path, candidate_masses
                        )
                        masses = candidate_masses
                        status = "reset all masses"
                    except Exception as exc:
                        status = f"model reset rejected: {exc}"
                elif key == ord("s"):
                    try:
                        save_link_masses(urdf_path, model, masses)
                        loaded_masses = masses.copy()
                        status = "saved masses to URDF"
                    except Exception as exc:
                        status = f"save failed: {exc}"
                elif key == ord("l"):
                    try:
                        candidate_all_masses = load_link_masses(urdf_path, model)
                        candidate_link_names = inverse_dynamics_link_names(
                            urdf_path,
                            model,
                            robot_config.base_link_name,
                            robot_config.eef_link_name,
                        )
                        if candidate_link_names != link_names:
                            raise ValueError("the inverse-dynamics link chain changed")
                        candidate_masses = OrderedDict(
                            (name, candidate_all_masses[name]) for name in link_names
                        )
                        candidate_solver = rebuild_solver(
                            robot_config, urdf_path, solver_path, candidate_masses
                        )
                        solver = candidate_solver
                        masses = candidate_masses
                        loaded_masses = candidate_masses.copy()
                        status = "loaded masses from URDF"
                    except Exception as exc:
                        status = f"load failed: {exc}"
                elif key == ord("p"):
                    paused = not paused
                key = stdscr.getch()

            state = controller.get_joint_state()
            gravity_torque = np.zeros(robot_config.joint_dof, dtype=np.float64)
            if gravity_enabled and not paused:
                gravity_torque = np.asarray(
                    solver.inverse_dynamics(
                        state.pos(),
                        np.zeros(robot_config.joint_dof, dtype=np.float64),
                        np.zeros(robot_config.joint_dof, dtype=np.float64),
                    ),
                    dtype=np.float64,
                )

            cmd = arx5.JointState(robot_config.joint_dof)
            cmd.torque()[:] = gravity_torque
            cmd.timestamp = controller.get_timestamp() + loop_dt
            controller.set_joint_cmd(cmd)

            draw(
                stdscr,
                model,
                interface,
                urdf_path,
                status,
                link_names,
                masses,
                loaded_masses,
                target_index,
                mass_step,
                step_scale,
                gravity_enabled,
                friction_enabled,
                paused,
                state,
                gravity_torque,
            )
            time.sleep(loop_dt)


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option("--loop-dt", default=0.02, show_default=True, type=float, help="Command loop period in seconds.")
@click.option(
    "--urdf-path",
    "--config-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="URDF used for loading and saving masses. Defaults to the selected model's URDF.",
)
@click.option("--mass-step", default=0.01, show_default=True, type=click.FloatRange(min=1e-9), help="Base mass adjustment in kg.")
@click.option("--min-mass", default=0.0, show_default=True, type=click.FloatRange(min=0.0), help="Minimum tunable link mass in kg.")
@click.option("--max-mass", default=10.0, show_default=True, type=click.FloatRange(min=0.0), help="Maximum tunable link mass in kg.")
def main(model, interface, loop_dt, urdf_path, mass_step, min_mass, max_mass):
    if loop_dt <= 0.0:
        raise click.ClickException("--loop-dt must be positive")
    if max_mass < min_mass:
        raise click.ClickException("--max-mass must be greater than or equal to --min-mass")

    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    if urdf_path is None:
        urdf_path = robot_config.urdf_path
    load_link_masses(urdf_path, model)
    robot_config.urdf_path = urdf_path

    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config
    )
    if not controller_config.joint_friction_compensation:
        raise click.ClickException(
            "The selected URDF has no enabled joint friction compensation terms."
        )
    controller_config.background_send_recv = True
    controller_config.gravity_compensation = False
    controller_config.friction_compensation = True

    controller = arx5.Arx5JointController(robot_config, controller_config, interface)
    controller.set_log_level(arx5.LogLevel.WARNING)

    gain = arx5.Gain(robot_config.joint_dof)
    gain.kp()[:] = 0.0
    gain.kd()[:] = 0.0
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    controller.set_gain(gain)
    controller.set_friction_compensation(True)

    try:
        curses.wrapper(
            run_tuner,
            controller,
            model,
            interface,
            loop_dt,
            urdf_path,
            mass_step,
            min_mass,
            max_mass,
        )
    finally:
        controller.set_gravity_compensation(False)
        controller.set_friction_compensation(False)
        zero_cmd = arx5.JointState(robot_config.joint_dof)
        zero_cmd.timestamp = controller.get_timestamp() + loop_dt
        controller.set_joint_cmd(zero_cmd)
        time.sleep(loop_dt * 2.0)
        controller.set_to_damping()


if __name__ == "__main__":
    main()
