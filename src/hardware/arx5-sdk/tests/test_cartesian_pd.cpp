#include "app/cartesian_pd.h"
#include "app/config.h"
#include "cartesian_ik.h"
#include "utils.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(const Eigen::MatrixXd &actual, const Eigen::MatrixXd &expected, double tolerance,
                  const std::string &message)
{
    const double error = (actual - expected).cwiseAbs().maxCoeff();
    if (error > tolerance)
        throw std::runtime_error(message + "; max error = " + std::to_string(error));
}

void require_near(double actual, double expected, double tolerance, const std::string &message)
{
    const double error = std::abs(actual - expected);
    if (error > tolerance)
        throw std::runtime_error(message + "; error = " + std::to_string(error));
}

Eigen::MatrixXd finite_difference_jacobian(const arx::CartesianKinematics &kinematics, const arx::VecDoF &joint_pos)
{
    constexpr double epsilon = 1e-6;
    Eigen::MatrixXd result(6, joint_pos.size());
    for (int i = 0; i < joint_pos.size(); ++i)
    {
        arx::VecDoF plus = joint_pos;
        arx::VecDoF minus = joint_pos;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        const KDL::Frame plus_frame = kinematics.frame(plus);
        const KDL::Frame minus_frame = kinematics.frame(minus);
        const KDL::Vector linear = (plus_frame.p - minus_frame.p) / (2.0 * epsilon);
        const KDL::Vector angular = (plus_frame.M * minus_frame.M.Inverse()).GetRot() / (2.0 * epsilon);
        result.col(i) << linear.x(), linear.y(), linear.z(), angular.x(), angular.y(), angular.z();
    }
    return result;
}

void test_jacobian(const std::string &model, const arx::VecDoF &joint_pos)
{
    const arx::RobotConfig config = arx::RobotConfigFactory::get_instance().get_config(model);
    arx::CartesianKinematics kinematics(config.urdf_path, config.joint_dof, config.base_link_name,
                                        config.eef_link_name);
    require_near(kinematics.jacobian(joint_pos), finite_difference_jacobian(kinematics, joint_pos), 2e-5,
                 model + " KDL Jacobian differs from centered finite-difference FK");
}

void test_pose_error_wraparound()
{
    constexpr double epsilon = 1e-4;
    const KDL::Frame desired(KDL::Rotation::RPY(0.0, 0.0, M_PI - epsilon));
    const KDL::Frame measured(KDL::Rotation::RPY(0.0, 0.0, -M_PI + epsilon));
    const arx::Pose6d error = arx::cartesian_pose_error(desired, measured);
    require(std::abs(error[5] + 2.0 * epsilon) < 1e-8, "SO(3) error did not take the short wraparound path");
    require(error.head<5>().norm() < 1e-10, "SO(3) wraparound introduced an error on another axis");
}

void test_pd_math()
{
    const Eigen::Matrix<double, 6, 6> jacobian = Eigen::Matrix<double, 6, 6>::Identity();
    const arx::CartesianPDGain gain((arx::Pose6d() << 10, 10, 10, 4, 4, 4).finished(),
                                    (arx::Pose6d() << 2, 2, 2, 1, 1, 1).finished());
    const arx::Pose6d limits = (arx::Pose6d() << 3, 3, 3, 2, 2, 2).finished();

    arx::Pose6d zero = arx::Pose6d::Zero();
    require(arx::compute_cartesian_pd_torque(jacobian, zero, zero, zero, gain, limits).isZero(1e-12),
            "Zero Cartesian error produced nonzero torque");

    arx::Pose6d pose_error = arx::Pose6d::Zero();
    pose_error[0] = 0.1;
    const arx::VecDoF restoring_torque =
        arx::compute_cartesian_pd_torque(jacobian, pose_error, zero, zero, gain, limits);
    require(restoring_torque[0] > 0.0, "Cartesian stiffness torque points away from the target");

    arx::Pose6d actual_twist = arx::Pose6d::Zero();
    actual_twist[1] = 0.5;
    const arx::VecDoF damping_torque =
        arx::compute_cartesian_pd_torque(jacobian, zero, zero, actual_twist, gain, limits);
    require(damping_torque[1] < 0.0, "Cartesian damping torque does not oppose measured velocity");

    pose_error.setConstant(100.0);
    arx::Pose6d wrench;
    arx::compute_cartesian_pd_torque(jacobian, pose_error, zero, zero, gain, limits, &wrench);
    require_near(wrench, limits, 1e-12, "Cartesian wrench was not clamped per axis");

    const arx::CartesianPDGain proportional_only(gain.kp, arx::Pose6d::Zero());
    arx::Pose6d proportional_wrench;
    const arx::VecDoF proportional_torque = arx::compute_cartesian_pd_torque(
        jacobian, arx::Pose6d::Constant(0.1), zero, actual_twist, proportional_only, limits,
        &proportional_wrench);
    require_near(proportional_torque, proportional_wrench, 1e-12,
                 "Proportional-only Cartesian mode included derivative torque");
}

void test_damped_cartesian_joint_velocity()
{
    const Eigen::Matrix<double, 6, 6> identity = Eigen::Matrix<double, 6, 6>::Identity();
    arx::Pose6d twist;
    twist << 1.0, -2.0, 0.5, 0.0, 3.0, -0.25;
    const arx::VecDoF limits = arx::VecDoF::Ones(6);
    const auto scaled = arx::compute_damped_cartesian_joint_velocity(identity, twist, limits, 0.01);
    require(scaled.valid, "Damped pseudoinverse rejected a finite full-rank Jacobian");
    require(scaled.scale < 1.0 && scaled.scale > 0.0, "Joint velocity limit did not uniformly scale the solution");
    require(scaled.limited.cwiseAbs().maxCoeff() <= 1.0 + 1e-12,
            "Limited pseudoinverse solution exceeded a joint velocity limit");
    require_near(scaled.limited, scaled.unlimited * scaled.scale, 1e-12,
                 "Pseudoinverse velocity limiting was not uniform");

    Eigen::Matrix<double, 6, 6> singular = identity;
    singular.row(5).setZero();
    const auto regularized = arx::compute_damped_cartesian_joint_velocity(singular, twist, limits, 0.01);
    require(regularized.valid && regularized.limited.allFinite(),
            "Damped pseudoinverse did not regularize a singular Jacobian");

    Eigen::Matrix<double, 6, 6> invalid = identity;
    invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
    require(!arx::compute_damped_cartesian_joint_velocity(invalid, twist, limits, 0.01).valid,
            "Pseudoinverse accepted a non-finite Jacobian");
}

void test_cartesian_error_clamp()
{
    arx::Pose6d error;
    error << 0.06, 0.08, 0.0, 0.3, 0.4, 0.0;
    const arx::Pose6d clamped = arx::clamp_cartesian_pose_error(error, 0.03, 0.25);
    require_near(clamped.head<3>().norm(), 0.03, 1e-12, "Translation error norm was not clamped");
    require_near(clamped.tail<3>().norm(), 0.25, 1e-12, "Rotation error norm was not clamped");
    require(arx::cartesian_tracking_error_exceeds(error, 0.03, 0.25),
            "Tracking guard missed an excessive Cartesian error");
    require(!arx::cartesian_tracking_error_recovered(error, 0.03, 0.25),
            "Tracking guard recovered without the required hysteresis");
    require(arx::cartesian_tracking_error_recovered(arx::Pose6d::Zero(), 0.03, 0.25),
            "Tracking guard did not recover at zero error");

    require(arx::cartesian_joint_limit_guard_triggered(-0.99, -0.03, -1.0, 1.0, 0.02, 0.02),
            "Joint-limit guard missed outward motion near the lower limit");
    require(!arx::cartesian_joint_limit_guard_triggered(-0.99, 0.03, -1.0, 1.0, 0.02, 0.02),
            "Joint-limit guard rejected inward recovery motion");
    require(arx::cartesian_joint_limit_guard_triggered(1.01, 0.0, -1.0, 1.0, 0.02, 0.02),
            "Joint-limit guard missed a hard-limit violation");
}

void test_cartesian_task_torque_slew_limit()
{
    const arx::VecDoF previous = arx::VecDoF::Zero(3);
    const arx::VecDoF requested = (arx::VecDoF(3) << 20.0, -20.0, 0.1).finished();
    const arx::VecDoF limits = (arx::VecDoF(3) << 10.0, 5.0, 2.0).finished();
    const arx::VecDoF first = arx::slew_limit_cartesian_task_torque(requested, previous, limits, 0.002, 0.02);
    require_near(first, (arx::VecDoF(3) << 1.0, -0.5, 0.1).finished(), 1e-12,
                 "Cartesian task torque slew did not enforce its per-tick limit");
    const arx::VecDoF second = arx::slew_limit_cartesian_task_torque(requested, first, limits, 0.002, 0.02);
    require_near(second, (arx::VecDoF(3) << 2.0, -1.0, 0.1).finished(), 1e-12,
                 "Cartesian task torque slew did not converge smoothly");
}

void test_cartesian_reference_interpolation()
{
    const arx::Pose6d velocity_limit = arx::Pose6d::Constant(2.0);
    arx::CartesianReferenceInterpolator reference(0.002, velocity_limit);
    const KDL::Frame start = arx::CartesianReferenceInterpolator::frame_from_pose(arx::Pose6d::Zero());
    reference.init_fixed(start, 1.0);
    require(reference.interpolate(1.5).twist.isZero(0.0), "Fixed Cartesian reference has nonzero twist");

    arx::Pose6d target_pose = arx::Pose6d::Zero();
    target_pose[0] = 0.2;
    target_pose[5] = 0.4;
    const KDL::Frame target = arx::CartesianReferenceInterpolator::frame_from_pose(target_pose);
    reference.override_waypoint(1.0, target, arx::Pose6d::Zero(), 2.0);
    const arx::CartesianReferenceSample midpoint = reference.interpolate(1.5);
    require_near(midpoint.frame.p.x(), 0.1, 1e-12, "Cartesian cubic translation midpoint is wrong");
    require_near(midpoint.twist[0], 0.3, 1e-12, "Cartesian cubic translation derivative is wrong");
    const arx::Pose6d midpoint_pose = arx::CartesianReferenceInterpolator::pose_from_frame(midpoint.frame);
    require_near(midpoint_pose[5], 0.2, 1e-10, "SO(3) cubic midpoint is wrong");
    require_near(midpoint.twist[5], 0.6, 1e-10, "SO(3) cubic angular velocity is wrong");
    require_near(arx::CartesianReferenceInterpolator::pose_from_frame(reference.interpolate(2.0).frame), target_pose,
                 1e-10, "Cartesian reference did not reach its endpoint");
    require(reference.interpolate(2.0).twist.isZero(1e-12), "Pose-only endpoint did not stop");

    const arx::CartesianReferenceSample before_retarget = reference.interpolate(1.6);
    arx::Pose6d second_target_pose = target_pose;
    second_target_pose[1] = 0.1;
    reference.override_waypoint(1.6, arx::CartesianReferenceInterpolator::frame_from_pose(second_target_pose),
                                arx::Pose6d::Zero(), 2.6);
    const arx::CartesianReferenceSample after_retarget = reference.interpolate(1.6);
    require_near(arx::cartesian_pose_error(after_retarget.frame, before_retarget.frame), arx::Pose6d::Zero(), 1e-12,
                 "Cartesian retarget changed pose at the handoff");
    require_near(after_retarget.twist, before_retarget.twist, 1e-12,
                 "Cartesian retarget changed twist at the handoff");
}

void test_100_hz_knots_interpolate_at_500_hz()
{
    constexpr double start_time = 1.0;
    constexpr double knot_dt = 0.01;
    constexpr double controller_dt = 0.002;

    arx::JointState joint_start(1);
    joint_start.timestamp = start_time;
    arx::JointState joint_knot_1(1);
    joint_knot_1.timestamp = start_time + knot_dt;
    joint_knot_1.pos[0] = 0.01;
    joint_knot_1.vel[0] = 1.0;
    arx::JointState joint_knot_2(1);
    joint_knot_2.timestamp = start_time + 2.0 * knot_dt;
    joint_knot_2.pos[0] = 0.02;
    joint_knot_2.vel[0] = 1.0;
    arx::JointStateInterpolator joint_reference(1, "linear");
    joint_reference.init_fixed(joint_start);
    joint_reference.override_traj(start_time, {joint_knot_1, joint_knot_2});

    joint_start.vel[0] = 1.0;
    arx::JointStateInterpolator cubic_joint_reference(1, "cubic");
    cubic_joint_reference.init(joint_start, joint_knot_1);

    arx::CartesianReferenceInterpolator cartesian_reference(
        controller_dt, arx::Pose6d::Constant(10.0));
    cartesian_reference.init_fixed(KDL::Frame::Identity(), start_time);
    std::vector<KDL::Frame> frames;
    std::vector<arx::Pose6d> twists;
    std::vector<double> timestamps;
    for (int knot = 1; knot <= 2; ++knot)
    {
        arx::Pose6d pose = arx::Pose6d::Zero();
        pose[0] = knot * knot_dt;
        pose[5] = knot * knot_dt;
        arx::Pose6d twist = arx::Pose6d::Zero();
        twist[0] = 1.0;
        twist[5] = 1.0;
        frames.push_back(arx::CartesianReferenceInterpolator::frame_from_pose(pose));
        twists.push_back(twist);
        timestamps.push_back(start_time + knot * knot_dt);
    }
    cartesian_reference.override_trajectory(start_time, frames, twists, timestamps);

    // A 100 Hz segment contains five 500 Hz control samples. Check every
    // interior controller tick between the two submitted knots.
    for (int tick = 1; tick < 5; ++tick)
    {
        const double timestamp = start_time + knot_dt + tick * controller_dt;
        const double expected = knot_dt + tick * controller_dt;
        require_near(joint_reference.interpolate(timestamp).pos[0], expected, 1e-12,
                     "Joint 100 Hz knots were not interpolated at a 500 Hz tick");
        const arx::Pose6d pose = arx::CartesianReferenceInterpolator::pose_from_frame(
            cartesian_reference.interpolate(timestamp).frame);
        require_near(pose[0], expected, 1e-12,
                     "Cartesian translation was not interpolated at a 500 Hz tick");
        require_near(pose[5], expected, 1e-10,
                     "Cartesian orientation was not interpolated at a 500 Hz tick");

        const double first_segment_timestamp = start_time + tick * controller_dt;
        const arx::JointState cubic_sample = cubic_joint_reference.interpolate(first_segment_timestamp);
        require_near(cubic_sample.pos[0], tick * controller_dt, 1e-12,
                     "Cubic joint interpolation omitted segment-duration scaling");
        require_near(cubic_sample.vel[0], 1.0, 1e-12,
                     "Cubic joint interpolation returned an incorrectly scaled velocity");
    }
}

void test_cartesian_reference_wrap_brake_and_velocity_limit()
{
    const arx::Pose6d velocity_limit = (arx::Pose6d() << 0.2, 0.2, 0.2, 0.5, 0.5, 0.5).finished();
    arx::CartesianReferenceInterpolator reference(0.002, velocity_limit, 0.05);
    arx::Pose6d start_pose = arx::Pose6d::Zero();
    start_pose[5] = M_PI - 0.1;
    arx::Pose6d end_pose = start_pose;
    end_pose[0] = 1.0;
    end_pose[5] = -M_PI + 0.1;
    reference.init_fixed(arx::CartesianReferenceInterpolator::frame_from_pose(start_pose), 1.0);
    arx::Pose6d final_twist = arx::Pose6d::Zero();
    final_twist[0] = 0.1;
    reference.override_waypoint(1.0, arx::CartesianReferenceInterpolator::frame_from_pose(end_pose), final_twist, 1.1);

    for (double time = 1.0; time <= 12.0; time += 0.002)
    {
        const arx::Pose6d twist = reference.interpolate(time).twist;
        require((twist.cwiseAbs().array() <= velocity_limit.array() + 1e-8).all(),
                "Retimed Cartesian reference exceeded ee_vel_max");
    }
    require(reference.interpolate(12.0).twist.isZero(1e-12), "Nonzero final twist did not brake to zero");
    const arx::Pose6d completed_pose =
        arx::CartesianReferenceInterpolator::pose_from_frame(reference.interpolate(12.0).frame);
    require(completed_pose[0] > end_pose[0], "Nonzero final twist brake did not produce bounded forward overshoot");
    const arx::Pose6d rotation_delta = arx::cartesian_pose_error(end_pose, start_pose);
    require_near(rotation_delta.tail<3>().norm(), 0.2, 1e-10, "Rotation interpolation did not use shortest path");
}

void test_cartesian_reference_analytic_twist_matches_finite_difference()
{
    arx::CartesianReferenceInterpolator reference(0.002, arx::Pose6d::Constant(10.0));
    arx::Pose6d start_pose;
    start_pose << 0.1, -0.2, 0.3, 0.3, -0.2, 0.4;
    arx::Pose6d end_pose;
    end_pose << 0.3, 0.1, 0.25, -0.1, 0.5, -0.3;
    arx::Pose6d terminal_twist;
    terminal_twist << 0.05, -0.02, 0.01, 0.1, -0.15, 0.08;
    reference.init_fixed(arx::CartesianReferenceInterpolator::frame_from_pose(start_pose), 1.0);
    reference.override_waypoint(1.0, arx::CartesianReferenceInterpolator::frame_from_pose(end_pose), terminal_twist,
                                2.0);

    constexpr double sample_time = 1.47;
    constexpr double epsilon = 1e-6;
    const arx::CartesianReferenceSample sample = reference.interpolate(sample_time);
    const KDL::Frame before = reference.interpolate(sample_time - epsilon).frame;
    const KDL::Frame after = reference.interpolate(sample_time + epsilon).frame;
    const KDL::Vector linear = (after.p - before.p) / (2.0 * epsilon);
    const KDL::Vector angular = (after.M * before.M.Inverse()).GetRot() / (2.0 * epsilon);
    arx::Pose6d finite_difference;
    finite_difference << linear.x(), linear.y(), linear.z(), angular.x(), angular.y(), angular.z();
    require_near(sample.twist, finite_difference, 2e-7,
                 "Analytic Cartesian reference twist differs from finite-difference pose");
    require_near(reference.interpolate(2.0).twist, terminal_twist, 1e-10,
                 "Explicit terminal twist was not met before braking");
    require(reference.interpolate(2.1).twist.isZero(1e-12), "Explicit terminal twist did not finish braking");
}

void test_defaults_and_validation()
{
    const arx::ControllerConfig config =
        arx::ControllerConfigFactory::get_instance().get_config("cartesian_controller", 6);
    require_near(config.default_cartesian_kp, arx::Pose6d::Zero(), 0.0,
                 "Unexpected default Cartesian kp");
    require_near(config.default_cartesian_kd, arx::Pose6d::Zero(), 0.0,
                 "Unexpected default Cartesian kd");
    require_near(config.cartesian_wrench_limit,
                 (arx::Pose6d() << 30, 30, 30, 5, 5, 5).finished(), 0.0,
                 "Unexpected Cartesian wrench limit");
    require(config.cartesian_joint_velocity_filter_cutoff_hz == 0.0,
            "Generic Cartesian configuration unexpectedly enables velocity filtering");
    require_near(config.cartesian_tracking_error_translation_limit_m, 0.10, 0.0,
                 "Unexpected Cartesian translation guard default");
    require_near(config.cartesian_tracking_error_rotation_limit_rad, 45.0 * M_PI / 180.0, 1e-15,
                 "Unexpected Cartesian rotation guard default");
    require_near(config.cartesian_task_torque_ramp_time_s, 0.02, 0.0,
                 "Unexpected Cartesian task-torque ramp default");

    bool rejected = false;
    try
    {
        arx::CartesianPDGain invalid((arx::Pose6d() << -1, 0, 0, 0, 0, 0).finished(), arx::Pose6d::Zero());
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected, "Negative Cartesian gain was accepted");

    rejected = false;
    try
    {
        arx::Pose6d invalid = arx::Pose6d::Zero();
        invalid[2] = std::numeric_limits<double>::quiet_NaN();
        arx::validate_cartesian_vector(invalid, "test");
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected, "Non-finite Cartesian value was accepted");

    rejected = false;
    try
    {
        arx::validate_cartesian_pd_model(7);
    }
    catch (const std::runtime_error &error)
    {
        rejected = std::string(error.what()).find("six-joint X5/L5") != std::string::npos;
    }
    require(rejected, "Seven-joint model did not produce a clear unsupported-model error");
}

void expect_invalid_filter_cutoff(double cutoff_hz, double controller_dt)
{
    bool rejected = false;
    try
    {
        arx::validate_cartesian_joint_velocity_filter_cutoff(cutoff_hz, controller_dt);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected, "Invalid Cartesian joint velocity filter cutoff was accepted");
}

void test_velocity_filter_initialization_bypass_gap_and_validation()
{
    constexpr double controller_dt = 0.002;
    arx::CartesianJointVelocityFilter filter(6, controller_dt, 50.0);
    const arx::VecDoF first = arx::VecDoF::Constant(6, 2.0);
    require_near(filter.update(first, 1.0), first, 0.0,
                 "Velocity filter did not initialize from its first measured sample");

    const arx::VecDoF second = arx::VecDoF::Constant(6, 4.0);
    const double alpha_50_hz = 1.0 - std::exp(-2.0 * std::acos(-1.0) * 50.0 * controller_dt);
    require_near(filter.update(second, 1.0 + controller_dt),
                 first + alpha_50_hz * (second - first), 1e-12,
                 "Velocity filter did not use the measured timestamp interval");

    const arx::VecDoF after_gap = arx::VecDoF::Constant(6, -3.0);
    require_near(filter.update(after_gap, 1.0 + 12.0 * controller_dt), after_gap, 0.0,
                 "Velocity filter did not reset after a gap over ten controller periods");

    const arx::VecDoF invalid_timestamp_sample = arx::VecDoF::Constant(6, 7.0);
    require_near(filter.update(invalid_timestamp_sample, std::numeric_limits<double>::quiet_NaN()),
                 invalid_timestamp_sample, 0.0, "Velocity filter did not reset on an invalid timestamp");
    const arx::VecDoF after_invalid_timestamp = arx::VecDoF::Constant(6, -1.0);
    require_near(filter.update(after_invalid_timestamp, 2.0), after_invalid_timestamp, 0.0,
                 "Velocity filter did not reinitialize after an invalid timestamp");

    filter.set_cutoff(0.0);
    const arx::VecDoF bypass_sample =
        (arx::VecDoF(6) << -4.0, -2.0, 0.0, 2.0, 4.0, 6.0).finished();
    require_near(filter.update(bypass_sample, 2.0 + controller_dt), bypass_sample, 0.0,
                 "Zero-cutoff velocity filter did not exactly bypass");

    filter.set_cutoff(10.0);
    const arx::VecDoF positive_sample = arx::VecDoF::Constant(6, 10.0);
    const double alpha_10_hz = 1.0 - std::exp(-2.0 * std::acos(-1.0) * 10.0 * controller_dt);
    const arx::VecDoF at_10_hz =
        bypass_sample + alpha_10_hz * (positive_sample - bypass_sample);
    require_near(filter.update(positive_sample, 2.0 + 2.0 * controller_dt), at_10_hz, 1e-12,
                 "Velocity filter did not start from the raw state synchronized during bypass");

    filter.set_cutoff(20.0);
    const double alpha_20_hz = 1.0 - std::exp(-2.0 * std::acos(-1.0) * 20.0 * controller_dt);
    require_near(filter.update(positive_sample, 2.0 + 3.0 * controller_dt),
                 at_10_hz + alpha_20_hz * (positive_sample - at_10_hz), 1e-12,
                 "Positive cutoff change discarded the existing filter state");

    expect_invalid_filter_cutoff(-1.0, controller_dt);
    expect_invalid_filter_cutoff(250.0, controller_dt);
    expect_invalid_filter_cutoff(std::numeric_limits<double>::infinity(), controller_dt);
    expect_invalid_filter_cutoff(50.0, 0.0);
}

void test_exact_50_hz_filter_coefficient_and_cartesian_damping()
{
    constexpr double controller_dt = 0.002;
    arx::CartesianJointVelocityFilter filter(6, controller_dt, 50.0);
    filter.update(arx::VecDoF::Zero(6), 0.0);
    const arx::VecDoF filtered = filter.update(arx::VecDoF::Ones(6), controller_dt);
    const double expected_alpha = 1.0 - std::exp(-2.0 * std::acos(-1.0) * 50.0 * controller_dt);
    require_near(filtered, arx::VecDoF::Constant(6, expected_alpha), 1e-12,
                 "50 Hz filter coefficient is incorrect at 500 Hz");
    require_near(expected_alpha, 0.4665119089088967, 1e-15,
                 "50 Hz filter coefficient no longer matches the specified value");

    const Eigen::Matrix<double, 6, 6> jacobian = Eigen::Matrix<double, 6, 6>::Identity();
    const arx::CartesianPDGain damping_only(arx::Pose6d::Zero(), arx::Pose6d::Ones());
    const arx::Pose6d limits = arx::Pose6d::Constant(100.0);
    const arx::VecDoF torque = arx::compute_cartesian_pd_torque(
        jacobian, arx::Pose6d::Zero(), arx::Pose6d::Zero(), filtered, damping_only, limits);
    require_near(torque, -filtered, 1e-12,
                 "Cartesian damping did not use the filtered measured velocity");
}

double measured_filter_gain(double signal_frequency_hz)
{
    constexpr double controller_dt = 0.002;
    constexpr int discard_samples = 1000;
    constexpr int measured_samples = 4000;
    arx::CartesianJointVelocityFilter filter(1, controller_dt, 50.0);
    double cosine_projection = 0.0;
    double sine_projection = 0.0;
    double cosine_energy = 0.0;
    double sine_energy = 0.0;
    for (int sample_index = 0; sample_index < discard_samples + measured_samples; ++sample_index)
    {
        const double timestamp = sample_index * controller_dt;
        const double phase = 2.0 * std::acos(-1.0) * signal_frequency_hz * timestamp;
        arx::VecDoF input(1);
        input[0] = std::cos(phase);
        const double output = filter.update(input, timestamp)[0];
        if (sample_index >= discard_samples)
        {
            const double cosine = std::cos(phase);
            const double sine = std::sin(phase);
            cosine_projection += output * cosine;
            sine_projection += output * sine;
            cosine_energy += cosine * cosine;
            sine_energy += sine * sine;
        }
    }

    const double cosine_component = cosine_projection / cosine_energy;
    if (sine_energy < 1e-12)
        return std::abs(cosine_component);
    const double sine_component = sine_projection / sine_energy;
    return std::sqrt(cosine_component * cosine_component + sine_component * sine_component);
}

double expected_discrete_filter_gain(double signal_frequency_hz)
{
    constexpr double controller_dt = 0.002;
    const double alpha = 1.0 - std::exp(-2.0 * std::acos(-1.0) * 50.0 * controller_dt);
    const double retained = 1.0 - alpha;
    const double omega = 2.0 * std::acos(-1.0) * signal_frequency_hz * controller_dt;
    return alpha / std::sqrt(1.0 + retained * retained - 2.0 * retained * std::cos(omega));
}

void test_velocity_filter_frequency_response()
{
    require(measured_filter_gain(5.0) >= 0.99, "50 Hz filter attenuated a 5 Hz velocity signal by over 1%");
    for (double frequency_hz : {50.0, 100.0, 150.0, 200.0, 250.0})
    {
        require_near(measured_filter_gain(frequency_hz), expected_discrete_filter_gain(frequency_hz), 1e-10,
                     "Velocity noise attenuation differs from the discrete one-pole transfer function at " +
                         std::to_string(frequency_hz) + " Hz");
    }
}

void test_l5_velocity_scaling_config()
{
    const arx::VecDoF expected =
        (arx::VecDoF(6) << 10.0 / 45.0, 10.0 / 45.0, 10.0 / 45.0,
         30.0 / 45.0, 30.0 / 45.0, 30.0 / 45.0)
            .finished();
    require(std::abs(arx::motor_protocol_velocity_command(arx::MotorType::DM_J4340, 2.0, 10.0 / 45.0) - 9.0) < 1e-12,
            "DM4340 command velocity did not apply Trossen L5 scaling");
    require(std::abs(arx::motor_protocol_velocity_command(arx::MotorType::DM_J4310, 2.0, 30.0 / 45.0) - 3.0) < 1e-12,
            "DM4310 command velocity did not apply Trossen L5 scaling");

    for (const std::string &model : {"L5", "Trossen_L5"})
    {
        const arx::RobotConfig config = arx::RobotConfigFactory::get_instance().get_config(model);
        require_near(config.joint_velocity_feedback_scale, expected, 1e-12,
                     model + " has incorrect joint velocity feedback scaling");
        require(std::abs(config.gripper_velocity_feedback_scale - 30.0 / 45.0) < 1e-12,
                model + " has incorrect gripper velocity feedback scaling");
    }

    const arx::RobotConfig x5 = arx::RobotConfigFactory::get_instance().get_config("X5");
    require_near(x5.joint_velocity_feedback_scale, arx::VecDoF::Ones(6), 0.0,
                 "L5 velocity scaling leaked into X5");
    require(x5.gripper_velocity_feedback_scale == 1.0, "L5 gripper velocity scaling leaked into X5");

    for (const std::string &model : {"L5", "Trossen_L5"})
    {
        const arx::RobotConfig robot_config = arx::RobotConfigFactory::get_instance().get_config(model);
        const arx::ControllerConfig controller_config =
            arx::ControllerConfigFactory::get_instance().get_config("cartesian_controller", robot_config);
        require(controller_config.cartesian_joint_velocity_filter_cutoff_hz == 0.0,
                model + " unexpectedly enables Cartesian velocity filtering by default");
        const arx::ControllerConfig joint_controller_config =
            arx::ControllerConfigFactory::get_instance().get_config("joint_controller", robot_config);
        require(joint_controller_config.cartesian_joint_velocity_filter_cutoff_hz == 0.0,
                model + " Cartesian velocity filter default leaked into joint control");
    }
    const arx::ControllerConfig x5_cartesian =
        arx::ControllerConfigFactory::get_instance().get_config("cartesian_controller", x5);
    require(x5_cartesian.cartesian_joint_velocity_filter_cutoff_hz == 0.0,
            "L5 Cartesian velocity filter default leaked into X5");
}

void test_ik_candidate_selection()
{
    const arx::VecDoF current = arx::VecDoF::Zero(6);
    const arx::VecDoF joint_min = arx::VecDoF::Constant(6, -1.0);
    const arx::VecDoF joint_max = arx::VecDoF::Constant(6, 1.0);

    arx::detail::IKCandidate failed_but_closest{-9, arx::VecDoF::Constant(6, 0.01)};
    arx::detail::IKCandidate farther_valid{0, arx::VecDoF::Constant(6, 0.3)};
    arx::detail::IKCandidate closest_valid{0, arx::VecDoF::Constant(6, 0.2)};
    const auto selected = arx::detail::select_closest_valid_ik_candidate(
        {failed_but_closest, farther_valid, closest_valid}, current, joint_min, joint_max);
    require(std::get<0>(selected) == 0, "IK selection returned a failed candidate despite valid alternatives");
    require_near(std::get<1>(selected), closest_valid.joint_position, 0.0,
                 "IK selection did not choose the closest valid candidate");

    arx::detail::IKCandidate at_limit{0, joint_max};
    const auto no_valid = arx::detail::select_closest_valid_ik_candidate(
        {failed_but_closest, at_limit}, current, joint_min, joint_max);
    require(std::get<0>(no_valid) != 0, "IK selection reported success when every candidate was invalid");
    require_near(std::get<1>(no_valid), current, 0.0,
                 "IK selection commanded a failed candidate instead of holding position");
}

} // namespace

int main()
{
    try
    {
        test_jacobian("X5", (arx::VecDoF(6) << 0.2, 0.5, 0.7, -0.3, 0.2, -0.4).finished());
        test_jacobian("L5", (arx::VecDoF(6) << -0.3, 0.6, 0.4, 0.2, -0.25, 0.35).finished());
        test_pose_error_wraparound();
        test_pd_math();
        test_damped_cartesian_joint_velocity();
        test_cartesian_error_clamp();
        test_cartesian_task_torque_slew_limit();
        test_cartesian_reference_interpolation();
        test_100_hz_knots_interpolate_at_500_hz();
        test_cartesian_reference_wrap_brake_and_velocity_limit();
        test_cartesian_reference_analytic_twist_matches_finite_difference();
        test_defaults_and_validation();
        test_velocity_filter_initialization_bypass_gap_and_validation();
        test_exact_50_hz_filter_coefficient_and_cartesian_damping();
        test_velocity_filter_frequency_response();
        test_l5_velocity_scaling_config();
        test_ik_candidate_selection();
        std::cout << "Cartesian PD unit tests passed" << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Cartesian PD unit test failure: " << error.what() << std::endl;
        return 1;
    }
}
