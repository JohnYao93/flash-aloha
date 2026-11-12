import shutil
import sys
import tempfile
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import numpy as np

import arx5_interface as arx5
from tune_gripper_friction_compensation import (
    REPO_DIR,
    FrictionTerms,
    apply_terms,
    body_friction_enabled,
    load_friction_config,
    save_friction_config,
    set_body_friction_enabled,
)


def test_gripper_friction_runtime_setter_is_bound():
    assert hasattr(arx5.Arx5JointController, "set_gravity_compensation")
    assert hasattr(arx5.Arx5CartesianController, "set_gravity_compensation")
    assert hasattr(arx5.Arx5JointController, "set_friction_compensation")
    assert hasattr(arx5.Arx5CartesianController, "set_friction_compensation")
    assert hasattr(arx5.Arx5JointController, "set_joint_friction_compensation")
    assert hasattr(arx5.Arx5CartesianController, "set_joint_friction_compensation")
    assert hasattr(arx5.Arx5JointController, "set_gripper_friction_compensation")
    assert hasattr(arx5.Arx5CartesianController, "set_gripper_friction_compensation")


def assert_friction_config_disabled(controller_config, joint_dof):
    assert controller_config.friction_compensation is False
    assert controller_config.joint_friction_compensation is False
    np.testing.assert_allclose(controller_config.joint_friction_transition_velocity, np.zeros(joint_dof))
    np.testing.assert_allclose(controller_config.joint_friction_constant_term, np.zeros(joint_dof))
    np.testing.assert_allclose(controller_config.joint_friction_coulomb_coef, np.zeros(joint_dof))
    np.testing.assert_allclose(controller_config.joint_friction_viscous_coef, np.zeros(joint_dof))
    assert controller_config.gripper_friction_compensation is False
    assert controller_config.gripper_friction_transition_velocity == 0.0
    assert controller_config.gripper_friction_constant_term == 0.0
    assert controller_config.gripper_friction_coulomb_coef == 0.0
    assert controller_config.gripper_friction_viscous_coef == 0.0


def test_gripper_friction_config_defaults_and_mutation():
    robot_config = arx5.RobotConfigFactory.get_instance().get_config("L5")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )

    assert_friction_config_disabled(controller_config, robot_config.joint_dof)

    controller_config.friction_compensation = True
    controller_config.joint_friction_compensation = True
    controller_config.joint_friction_transition_velocity = np.full(robot_config.joint_dof, 0.2)
    controller_config.joint_friction_constant_term = np.full(robot_config.joint_dof, 0.1)
    controller_config.joint_friction_coulomb_coef = np.full(robot_config.joint_dof, 0.3)
    controller_config.joint_friction_viscous_coef = np.full(robot_config.joint_dof, 0.5)
    controller_config.gripper_friction_compensation = True
    controller_config.gripper_friction_transition_velocity = 0.2
    controller_config.gripper_friction_constant_term = 0.1
    controller_config.gripper_friction_coulomb_coef = 0.3
    controller_config.gripper_friction_viscous_coef = 0.5

    assert controller_config.friction_compensation is True
    assert controller_config.joint_friction_compensation is True
    np.testing.assert_allclose(controller_config.joint_friction_transition_velocity, np.full(robot_config.joint_dof, 0.2))
    np.testing.assert_allclose(controller_config.joint_friction_constant_term, np.full(robot_config.joint_dof, 0.1))
    np.testing.assert_allclose(controller_config.joint_friction_coulomb_coef, np.full(robot_config.joint_dof, 0.3))
    np.testing.assert_allclose(controller_config.joint_friction_viscous_coef, np.full(robot_config.joint_dof, 0.5))
    assert controller_config.gripper_friction_compensation is True
    assert controller_config.gripper_friction_transition_velocity == 0.2
    assert controller_config.gripper_friction_constant_term == 0.1
    assert controller_config.gripper_friction_coulomb_coef == 0.3
    assert controller_config.gripper_friction_viscous_coef == 0.5


def test_l5_model_aware_config_enables_saved_friction_defaults():
    for model, joint_3_coulomb in (("L5", 0.15), ("Trossen_L5", 0.1)):
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config
        )

        assert controller_config.friction_compensation is True
        assert controller_config.joint_friction_compensation is True
        np.testing.assert_allclose(
            controller_config.joint_friction_transition_velocity,
            np.array([0.01, 0.01, 0.01, 0.015, 0.015, 0.015]),
        )
        np.testing.assert_allclose(
            controller_config.joint_friction_constant_term,
            np.array([0.0, 0.0, 3.46944695195e-18, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            controller_config.joint_friction_coulomb_coef,
            np.array([0.1, 0.1, joint_3_coulomb, 0.02, 0.02, 0.02]),
        )
        np.testing.assert_allclose(
            controller_config.joint_friction_viscous_coef,
            np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0]),
        )
        assert controller_config.gripper_friction_compensation is True
        assert controller_config.gripper_friction_transition_velocity == 0.001
        assert controller_config.gripper_friction_constant_term == 0.0
        assert controller_config.gripper_friction_coulomb_coef == 0.05
        assert controller_config.gripper_friction_viscous_coef == 0.02


def test_x5_model_aware_config_keeps_friction_disabled():
    robot_config = arx5.RobotConfigFactory.get_instance().get_config("X5")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", robot_config)

    assert_friction_config_disabled(controller_config, robot_config.joint_dof)


def test_gripper_friction_config_available_on_cartesian_controller_config():
    robot_config = arx5.RobotConfigFactory.get_instance().get_config("L5")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config.joint_dof
    )

    controller_config.joint_friction_compensation = True
    controller_config.joint_friction_transition_velocity = np.full(robot_config.joint_dof, 0.2)
    controller_config.joint_friction_constant_term = np.full(robot_config.joint_dof, 0.1)
    controller_config.joint_friction_coulomb_coef = np.full(robot_config.joint_dof, 0.3)
    controller_config.joint_friction_viscous_coef = np.full(robot_config.joint_dof, 0.5)
    controller_config.gripper_friction_compensation = True
    controller_config.gripper_friction_transition_velocity = 0.05

    assert controller_config.gripper_friction_compensation is True
    assert controller_config.gripper_friction_transition_velocity == 0.05


def friction_block_outside_text(text):
    tag = "arx5_friction_compensation"
    start = text.find(f"<{tag}")
    line_start = text.rfind("\n", 0, start) + 1
    end = text.find(f"</{tag}>", start) + len(f"</{tag}>")
    return text[:line_start], text[end:]


def test_tuner_body_and_gripper_friction_can_be_disabled_independently():
    gripper_terms = FrictionTerms(enabled=True, coulomb_coef=0.2)
    joint_terms = [FrictionTerms(enabled=True, coulomb_coef=0.1) for _ in range(6)]

    set_body_friction_enabled(joint_terms, False)
    assert body_friction_enabled(joint_terms) is False
    assert gripper_terms.enabled is True
    assert all(terms.coulomb_coef == 0.1 for terms in joint_terms)

    set_body_friction_enabled(joint_terms, True)
    gripper_terms.enabled = False
    assert body_friction_enabled(joint_terms) is True
    assert gripper_terms.enabled is False
    assert gripper_terms.coulomb_coef == 0.2


def test_tuner_applies_body_and_gripper_enable_flags_independently():
    class RecordingController:
        def set_gripper_friction_compensation(self, *args):
            self.gripper_args = args

        def set_joint_friction_compensation(self, *args):
            self.joint_args = args

        def set_friction_compensation(self, enabled):
            self.master_enabled = enabled

    controller = RecordingController()
    gripper_terms = FrictionTerms(enabled=True, transition_velocity=0.03, coulomb_coef=0.2)
    joint_terms = [FrictionTerms(enabled=False, transition_velocity=0.04, coulomb_coef=0.1) for _ in range(6)]

    apply_terms(controller, gripper_terms, joint_terms, True)
    assert controller.gripper_args[0] is True
    assert controller.joint_args[0] is False
    assert controller.master_enabled is True

    gripper_terms.enabled = False
    set_body_friction_enabled(joint_terms, True)
    apply_terms(controller, gripper_terms, joint_terms, True)
    assert controller.gripper_args[0] is False
    assert controller.joint_args[0] is True
    assert controller.master_enabled is True


def test_tuner_urdf_save_load_round_trip():
    source_urdf = Path(REPO_DIR) / "models" / "L5.urdf"
    with tempfile.TemporaryDirectory() as temp_dir:
        test_urdf = Path(temp_dir) / "L5.urdf"
        shutil.copy2(source_urdf, test_urdf)
        original_text = test_urdf.read_text()

        enabled, gripper_terms, joint_terms = load_friction_config(test_urdf, "L5", 6)
        assert enabled is True
        joint_terms[2].coulomb_coef = 0.234
        gripper_terms.viscous_coef = 0.056
        save_friction_config(test_urdf, "L5", False, gripper_terms, joint_terms)

        loaded_enabled, loaded_gripper, loaded_joints = load_friction_config(test_urdf, "L5", 6)
        assert loaded_enabled is False
        assert loaded_joints[2].coulomb_coef == 0.234
        assert loaded_gripper.viscous_coef == 0.056
        assert friction_block_outside_text(test_urdf.read_text()) == friction_block_outside_text(original_text)

        unchanged = test_urdf.read_text()
        loaded_joints[0].transition_velocity = 0.0
        try:
            save_friction_config(test_urdf, "L5", True, loaded_gripper, loaded_joints)
        except ValueError:
            pass
        else:
            raise AssertionError("saving enabled zero-transition terms should fail")
        assert test_urdf.read_text() == unchanged


if __name__ == "__main__":
    test_gripper_friction_runtime_setter_is_bound()
    test_gripper_friction_config_defaults_and_mutation()
    test_l5_model_aware_config_enables_saved_friction_defaults()
    test_x5_model_aware_config_keeps_friction_disabled()
    test_gripper_friction_config_available_on_cartesian_controller_config()
    test_tuner_body_and_gripper_friction_can_be_disabled_independently()
    test_tuner_applies_body_and_gripper_enable_flags_independently()
    test_tuner_urdf_save_load_round_trip()
    print("friction compensation Python checks passed")
