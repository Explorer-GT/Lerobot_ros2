#include <moveit/kinematics_base/kinematics_base.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_monitor/current_state_monitor.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/utils/moveit_error_code.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
using MoveGroup = moveit::planning_interface::MoveGroupInterface;
using JointTarget = std::map<std::string, double>;
const std::vector<std::string> kArmJoints{
  "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"};
const std::vector<std::string> kGripperJoints{"gripper"};
constexpr double kMaxStateAge = 1.0;
constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

enum class TargetType {Home, Gripper, Tcp};

struct Step
{
  const char * stage;
  const char * pose;
  TargetType type;
  const char * stop_name;
};

// Display the requested sequence; unique stop names distinguish repeated stages.
constexpr std::array<Step, 11> kSequence{{
  {"HOME", "home", TargetType::Home, "HOME"},
  {"OPEN_GRIPPER", "gripper_open", TargetType::Gripper, "OPEN_GRIPPER"},
  {"PRE_GRASP", "pre_grasp", TargetType::Tcp, "PRE_GRASP"},
  {"GRASP", "grasp", TargetType::Tcp, "GRASP"},
  {"CLOSE_GRIPPER", "gripper_closed", TargetType::Gripper, "CLOSE_GRIPPER"},
  {"LIFT", "lift", TargetType::Tcp, "LIFT"},
  {"TRANSFER", "transfer", TargetType::Tcp, "TRANSFER"},
  {"PLACE", "place", TargetType::Tcp, "PLACE"},
  {"OPEN_GRIPPER", "gripper_open", TargetType::Gripper, "RELEASE_GRIPPER"},
  {"RETREAT", "retreat", TargetType::Tcp, "RETREAT"},
  {"HOME", "home", TargetType::Home, "FINAL_HOME"},
}};

template<typename T>
void defaultParameter(
  const rclcpp::Node::SharedPtr & node, const std::string & name, const T & value)
{
  if (!node->has_parameter(name)) {
    node->declare_parameter<T>(name, value);
  }
}

void declareDefaults(const rclcpp::Node::SharedPtr & node)
{
  defaultParameter(node, "execute", false);
  // The supplied qualification YAML overrides this to PRE_GRASP.
  defaultParameter(node, "stop_after_stage", std::string("FULL_CYCLE"));
  defaultParameter(node, "target_link", std::string("tcp_link"));
  defaultParameter(node, "target_frame", std::string("world"));
  const JointTarget defaults{
    {"velocity_scaling", 0.1}, {"acceleration_scaling", 0.1},
    {"planning_time", 5.0}, {"state_timeout", 5.0}, {"server_timeout", 10.0},
    {"max_fk_position_error", 0.002}, {"max_reference_joint_delta", 0.1},
    {"trajectory_margin", 0.15},
    {"home_joint_tolerance", 0.001}, {"gripper_joint_tolerance", 0.001},
    {"trajectory_start_tolerance", 0.01}, {"trajectory_goal_tolerance", 0.001},
    {"max_gripper_joint_delta", 2.0},
    {"stage_max_joint_delta.home", 2.0}, {"stage_max_joint_delta.pre_grasp", 2.0},
    {"stage_max_joint_delta.grasp", 0.5}, {"stage_max_joint_delta.lift", 0.5},
    {"stage_max_joint_delta.transfer", 0.6}, {"stage_max_joint_delta.place", 0.5},
    {"stage_max_joint_delta.retreat", 0.5}};
  for (const auto & [name, value] : defaults) {
    defaultParameter(node, name, value);
  }
}

double positiveParameter(const rclcpp::Node::SharedPtr & node, const std::string & name)
{
  const double value = node->get_parameter(name).as_double();
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(name + " must be finite and positive");
  }
  return value;
}

class TcpPickPlace
{
public:
  explicit TcpPickPlace(const rclcpp::Node::SharedPtr & node)
  : node_(node),
    execute_(node->get_parameter("execute").as_bool()),
    target_link_(node->get_parameter("target_link").as_string()),
    target_frame_(node->get_parameter("target_frame").as_string()),
    stop_after_stage_(node->get_parameter("stop_after_stage").as_string()),
    state_timeout_(positiveParameter(node, "state_timeout")),
    max_fk_position_error_(positiveParameter(node, "max_fk_position_error")),
    max_reference_joint_delta_(positiveParameter(node, "max_reference_joint_delta")),
    trajectory_margin_(positiveParameter(node, "trajectory_margin")),
    home_joint_tolerance_(positiveParameter(node, "home_joint_tolerance")),
    gripper_joint_tolerance_(positiveParameter(node, "gripper_joint_tolerance")),
    trajectory_start_tolerance_(positiveParameter(node, "trajectory_start_tolerance")),
    trajectory_goal_tolerance_(positiveParameter(node, "trajectory_goal_tolerance")),
    max_gripper_joint_delta_(positiveParameter(node, "max_gripper_joint_delta")),
    tf_buffer_(std::make_shared<tf2_ros::Buffer>(node->get_clock())),
    arm_(node, "arm", tf_buffer_,
      rclcpp::Duration::from_seconds(positiveParameter(node, "server_timeout"))),
    gripper_(node, "gripper", tf_buffer_,
      rclcpp::Duration::from_seconds(positiveParameter(node, "server_timeout"))),
    state_monitor_(node, arm_.getRobotModel(), tf_buffer_, false)
  {
  }

  bool run()
  {
    try {
      stage_ = "PREPARE";
      prepare();
      if (!execute_) {
        RCLCPP_WARN(node_->get_logger(),
          "PLAN ONLY: each plan uses the actual measured state. HOME/gripper plans are not "
          "applied; this is not a chained-cycle validation. Stop no later than the first TCP stage.");
      }
      for (std::size_t index = 0; index < kSequence.size(); ++index) {
        const auto & step = kSequence[index];
        stage_ = std::to_string(index + 1) + "/11 " + step.stage;
        RCLCPP_INFO(node_->get_logger(), "[START] %s pose=%s stop_name=%s execute=%s",
          stage_.c_str(), step.pose, step.stop_name, execute_ ? "true" : "false");
        runStep(step);
        if (!execute_) {
          RCLCPP_INFO(node_->get_logger(), "[PLANNED ONLY] %s; target was not executed",
            stage_.c_str());
          if (index == stop_index_ || step.type == TargetType::Tcp) {
            RCLCPP_INFO(node_->get_logger(),
              "[PLAN ONLY STOP] last planned stage=%s, requested stop=%s. "
              "No subsequent stage or execution requested; preceding targets were not applied.",
              step.stop_name, stop_after_stage_.c_str());
            return true;
          }
        } else {
          RCLCPP_INFO(node_->get_logger(), "[DONE] %s; fresh final state verified", stage_.c_str());
          if (index == stop_index_) {
            if (index + 1 == kSequence.size()) {
              RCLCPP_INFO(node_->get_logger(), "TCP Pick & Place cycle complete (11/11 steps).");
            } else {
              RCLCPP_INFO(node_->get_logger(),
                "[QUALIFICATION STOP] %s succeeded. Next stage will not start.", step.stop_name);
            }
            return true;
          }
        }
      }
      throw std::runtime_error("State machine ended without its configured stop");
    } catch (const std::exception & error) {
      RCLCPP_ERROR(node_->get_logger(), "[FAILED] stage=%s: %s. State machine halted.",
        stage_.c_str(), error.what());
      // No retries, automatic release, fallback joint pose, or return home.
      // Plan-only mode has no execution to stop and sends no stop commands.
      if (execute_ && rclcpp::ok()) {
        for (auto * group : {&arm_, &gripper_}) {
          try {
            group->stop();
          } catch (const std::exception & stop_error) {
            RCLCPP_ERROR(node_->get_logger(), "Stop request for %s failed: %s",
              group->getName().c_str(), stop_error.what());
          }
        }
      }
      return false;
    }
  }

private:
  const moveit::core::JointModelGroup * validateGroup(
    MoveGroup & group, const std::vector<std::string> & expected) const
  {
    const auto * model_group = group.getRobotModel()->getJointModelGroup(group.getName());
    if (!model_group) {
      throw std::runtime_error("Missing planning group " + group.getName());
    }
    const auto & actual = model_group->getVariableNames();
    if (actual.size() != expected.size() ||
      std::set<std::string>(actual.begin(), actual.end()) !=
      std::set<std::string>(expected.begin(), expected.end()))
    {
      throw std::runtime_error("Unexpected joint variables in group " + group.getName());
    }
    return model_group;
  }

  void prepare()
  {
    arm_group_ = validateGroup(arm_, kArmJoints);
    validateGroup(gripper_, kGripperJoints);
    std::set<std::string> expected(kArmJoints.begin(), kArmJoints.end());
    expected.insert("gripper");
    const auto & variables = arm_.getRobotModel()->getVariableNames();
    if (variables.size() != 6 ||
      std::set<std::string>(variables.begin(), variables.end()) != expected)
    {
      throw std::runtime_error("RobotModel must contain exactly all six expected joint variables");
    }
    if (!arm_group_->isChain() || target_link_ != "tcp_link" || target_frame_ != "world" ||
      arm_.getRobotModel()->getModelFrame() != target_frame_ ||
      !arm_.getRobotModel()->hasLinkModel(target_link_))
    {
      throw std::runtime_error("Require arm chain, tcp_link, and world RobotModel target frame");
    }
    const auto description = node_->get_parameter("robot_description").as_string();
    if (description.find("mock_components/GenericSystem") == std::string::npos) {
      throw std::runtime_error("Client model must use mock_components/GenericSystem fake hardware");
    }
    const std::string prefix = "robot_description_kinematics.arm.";
    if (node_->get_parameter(prefix + "kinematics_solver").as_string() !=
      "kdl_kinematics_plugin/KDLKinematicsPlugin" ||
      !node_->get_parameter(prefix + "position_only_ik").as_bool())
    {
      throw std::runtime_error("Require KDL with position_only_ik=true");
    }
    const auto solver = arm_group_->getSolverInstance();
    if (!solver || solver->getTipFrames().size() != 1 ||
      solver->getTipFrames().front() != target_link_ ||
      !std::isfinite(arm_group_->getDefaultIKTimeout()) || arm_group_->getDefaultIKTimeout() <= 0.0)
    {
      throw std::runtime_error("Missing TCP IK solver, mismatched tip, or invalid IK timeout");
    }
    const double velocity = positiveParameter(node_, "velocity_scaling");
    const double acceleration = positiveParameter(node_, "acceleration_scaling");
    if (velocity > 0.1 || acceleration > 0.1) {
      throw std::runtime_error("Fake-hardware qualification requires scaling <= 0.1");
    }
    const double planning_time = positiveParameter(node_, "planning_time");
    for (auto * group : {&arm_, &gripper_}) {
      group->setMaxVelocityScalingFactor(velocity);
      group->setMaxAccelerationScalingFactor(acceleration);
      group->setPlanningTime(planning_time);
      group->setGoalJointTolerance(trajectory_goal_tolerance_);
      group->setNumPlanningAttempts(1);
      group->allowReplanning(false);
    }

    stop_index_ = kSequence.size() - 1;
    if (stop_after_stage_ != "FULL_CYCLE") {
      const auto found = std::find_if(kSequence.begin(), kSequence.end(), [this](const Step & step) {
          return stop_after_stage_ == step.stop_name;
        });
      if (found == kSequence.end()) {
        throw std::runtime_error("Unknown stop_after_stage: " + stop_after_stage_);
      }
      stop_index_ = static_cast<std::size_t>(std::distance(kSequence.begin(), found));
    }

    // Validate every target and taught reference before any movement. Runtime
    // IK copies the fresh measured state, then seeds only its arm with the reference.
    joint_targets_.emplace("home", readJointTarget("joint_poses.home", kArmJoints));
    joint_targets_.emplace("gripper_open", readJointTarget("joint_poses.gripper_open", kGripperJoints));
    joint_targets_.emplace("gripper_closed", readJointTarget("joint_poses.gripper_closed", kGripperJoints));
    for (const auto & step : kSequence) {
      if (step.type == TargetType::Tcp) {
        tcp_targets_.emplace(step.pose, readPosition("tcp_poses." + std::string(step.pose)));
        reference_targets_.emplace(step.pose,
          readJointTarget("reference_joint_poses." + std::string(step.pose), kArmJoints));
      }
      if (step.type != TargetType::Gripper && stage_limits_.count(step.pose) == 0) {
        stage_limits_.emplace(step.pose,
          positiveParameter(node_, "stage_max_joint_delta." + std::string(step.pose)));
        RCLCPP_INFO(node_->get_logger(), "Stage %s maximum joint displacement=%.6f rad",
          step.pose, stage_limits_.at(step.pose));
      }
    }
    home_fk_reference_ = readPosition("home_fk_reference");
    RCLCPP_INFO(node_->get_logger(),
      "Maximum taught-reference deviation per arm joint=%.6f rad (%.6f deg); "
      "reference seeds are never direct commands or the planning start state",
      max_reference_joint_delta_, max_reference_joint_delta_ * kRadiansToDegrees);
    RCLCPP_WARN(node_->get_logger(),
      "Nominal CAD-derived TCP at gripper q_ref=0; not a calibrated physical grasp center. "
      "Position-only IK does not constrain orientation or a straight TCP path. "
      "Verified taught PRE_GRASP wrist_flex has only about 0.114 degrees of upper-limit headroom.");
    RCLCPP_INFO(node_->get_logger(),
      "frame=%s target_link=%s IK_base=%s execute=%s stop_after_stage=%s. "
      "Requires the separately running fake-hardware demo stack.",
      target_frame_.c_str(), target_link_.c_str(), solver->getBaseFrame().c_str(),
      execute_ ? "true" : "false", stop_after_stage_.c_str());
    state_monitor_.startStateMonitor("/joint_states");
    const auto initial = measuredState();
    for (const auto & joint : variables) {
      RCLCPP_INFO(node_->get_logger(), "Startup measured %s=%.9f rad",
        joint.c_str(), initial->getVariablePosition(joint));
    }
  }

  JointTarget readJointTarget(
    const std::string & prefix, const std::vector<std::string> & joints) const
  {
    JointTarget values;
    node_->get_parameters(prefix, values);
    if (values.size() != joints.size()) {
      throw std::runtime_error("Wrong/missing joint keys for " + prefix);
    }
    for (const auto & joint : joints) {
      if (values.count(joint) != 1) {
        throw std::runtime_error("Missing " + joint + " in " + prefix);
      }
      validatePosition(joint, values.at(joint));
    }
    return values;
  }

  Eigen::Vector3d readPosition(const std::string & prefix) const
  {
    JointTarget values;
    node_->get_parameters(prefix, values);
    if (values.size() != 3 || values.count("x") != 1 || values.count("y") != 1 ||
      values.count("z") != 1)
    {
      throw std::runtime_error("Require exactly x/y/z keys for " + prefix);
    }
    const Eigen::Vector3d position(values.at("x"), values.at("y"), values.at("z"));
    if (!position.allFinite()) {
      throw std::runtime_error("Non-finite XYZ for " + prefix);
    }
    return position;
  }

  moveit::core::RobotStatePtr measuredState()
  {
    const auto requested_time = node_->now();
    auto next_update = requested_time;
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration<double>(state_timeout_);
    while (rclcpp::ok()) {
      const double remaining =
        std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count();
      if (remaining <= 0.0 || !state_monitor_.waitForCurrentState(next_update, remaining)) {
        break;
      }
      const auto oldest_allowed = std::max(
        requested_time, node_->now() - rclcpp::Duration::from_seconds(kMaxStateAge));
      if (state_monitor_.haveCompleteState(oldest_allowed)) {
        auto state = state_monitor_.getCurrentState();
        for (const auto & joint : state->getVariableNames()) {
          validatePosition(joint, state->getVariablePosition(joint));
        }
        if (!state->satisfiesBounds()) {
          throw std::runtime_error("Measured state violates MoveIt joint bounds");
        }
        state->update();
        return state;
      }
      next_update = state_monitor_.getCurrentStateTime() + rclcpp::Duration(0, 1);
    }
    throw std::runtime_error("Missing/stale /joint_states: all six joints require fresh updates");
  }

  void validatePosition(const std::string & joint, double value) const
  {
    if (!std::isfinite(value)) {
      throw std::runtime_error("Non-finite joint position for " + joint);
    }
    const auto & bounds = arm_.getRobotModel()->getVariableBounds(joint);
    if (bounds.position_bounded_ &&
      (value < bounds.min_position_ || value > bounds.max_position_))
    {
      throw std::runtime_error("Joint limit violation for " + joint);
    }
  }

  JointTarget positions(
    const moveit::core::RobotState & state, const std::vector<std::string> & joints) const
  {
    JointTarget values;
    for (const auto & joint : joints) {
      values.emplace(joint, state.getVariablePosition(joint));
    }
    return values;
  }

  void validateTarget(
    const JointTarget & current, const JointTarget & target, double maximum) const
  {
    for (const auto & [joint, value] : target) {
      validatePosition(joint, value);
      const double delta = std::abs(value - current.at(joint));
      RCLCPP_INFO(node_->get_logger(), "Joint %s measured=%.9f target=%.9f delta=%.9f rad max=%.6f",
        joint.c_str(), current.at(joint), value, delta, maximum);
      if (!std::isfinite(delta) || delta > maximum) {
        throw std::runtime_error("Stage joint displacement exceeds configured limit for " + joint);
      }
    }
  }

  void verifyJoints(
    const moveit::core::RobotState & state, const JointTarget & target,
    double tolerance, const char * label) const
  {
    for (const auto & [joint, value] : target) {
      const double actual = state.getVariablePosition(joint);
      const double error = std::abs(actual - value);
      RCLCPP_INFO(node_->get_logger(), "%s %s actual=%.9f target=%.9f error=%.9f rad tolerance=%.6f",
        label, joint.c_str(), actual, value, error, tolerance);
      if (!std::isfinite(error) || error > tolerance) {
        throw std::runtime_error(std::string(label) + ": joint mismatch for " + joint);
      }
    }
  }

  void validateReferencePosture(
    const JointTarget & current, const JointTarget & reference,
    const JointTarget & solution, double stage_maximum) const
  {
    // Log all five comparisons before rejecting any joint so a wrong branch
    // remains diagnosable even when its first failing joint halts this stage.
    for (const auto & joint : kArmJoints) {
      const double measured = current.at(joint);
      const double taught = reference.at(joint);
      const double result = solution.at(joint);
      const double current_delta = std::abs(result - measured);
      const double reference_delta = std::abs(result - taught);
      RCLCPP_INFO(node_->get_logger(),
        "IK posture %s: measured=%.9f rad (%.6f deg), taught_reference=%.9f rad (%.6f deg), "
        "solution=%.9f rad (%.6f deg)",
        joint.c_str(), measured, measured * kRadiansToDegrees,
        taught, taught * kRadiansToDegrees, result, result * kRadiansToDegrees);
      RCLCPP_INFO(node_->get_logger(),
        "IK differences %s: measured_to_solution=%.9f rad (%.6f deg), "
        "stage_max=%.6f rad (%.6f deg); reference_to_solution=%.9f rad (%.6f deg), "
        "reference_max=%.6f rad (%.6f deg)",
        joint.c_str(), current_delta, current_delta * kRadiansToDegrees,
        stage_maximum, stage_maximum * kRadiansToDegrees,
        reference_delta, reference_delta * kRadiansToDegrees,
        max_reference_joint_delta_, max_reference_joint_delta_ * kRadiansToDegrees);
    }
    for (const auto & joint : kArmJoints) {
      const double delta = std::abs(solution.at(joint) - reference.at(joint));
      if (!std::isfinite(delta) || delta > max_reference_joint_delta_) {
        throw std::runtime_error("IK taught-reference deviation exceeds configured limit for " + joint);
      }
    }
    // This independent measured-current check remains the physical transition
    // limit; proximity to the taught reference does not replace it.
    validateTarget(current, solution, stage_maximum);
  }

  Eigen::Isometry3d logTcp(moveit::core::RobotState & state, const char * label) const
  {
    state.update();
    const Eigen::Isometry3d transform = state.getGlobalLinkTransform(target_link_);
    if (!transform.matrix().allFinite()) {
      throw std::runtime_error("Non-finite TCP FK transform");
    }
    Eigen::Quaterniond quaternion(transform.linear());
    if (!quaternion.coeffs().allFinite() || quaternion.norm() <= 0.0) {
      throw std::runtime_error("Invalid TCP quaternion");
    }
    quaternion.normalize();
    if (quaternion.w() < 0.0) {
      quaternion.coeffs() *= -1.0;
    }
    const Eigen::Matrix3d rotation = transform.linear();
    // Fixed XYZ RPY: R = Rz(yaw) * Ry(pitch) * Rx(roll); diagnostics only.
    const double roll = std::atan2(rotation(2, 1), rotation(2, 2)) * kRadiansToDegrees;
    const double pitch = std::atan2(-rotation(2, 0),
      std::hypot(rotation(0, 0), rotation(1, 0))) * kRadiansToDegrees;
    const double yaw = std::atan2(rotation(1, 0), rotation(0, 0)) * kRadiansToDegrees;
    const auto & xyz = transform.translation();
    RCLCPP_INFO(node_->get_logger(),
      "%s TCP frame=%s XYZ=(%.12f, %.12f, %.12f) m quaternion_xyzw=(%.9f, %.9f, %.9f, %.9f) "
      "RPY=(%.6f, %.6f, %.6f) deg; orientation unconstrained",
      label, target_frame_.c_str(), xyz.x(), xyz.y(), xyz.z(),
      quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w(), roll, pitch, yaw);
    return transform;
  }

  void verifyTcp(
    moveit::core::RobotState & state, const Eigen::Vector3d & target, const char * label) const
  {
    const double error = (logTcp(state, label).translation() - target).norm();
    RCLCPP_INFO(node_->get_logger(), "%s TCP target=(%.12f, %.12f, %.12f) error=%.9f m max=%.6f",
      label, target.x(), target.y(), target.z(), error, max_fk_position_error_);
    if (!std::isfinite(error) || error > max_fk_position_error_) {
      throw std::runtime_error(std::string(label) + ": TCP position error exceeds configured limit");
    }
  }

  JointTarget solveTcp(
    const Step & step, const moveit::core::RobotStatePtr & measured, const JointTarget & current)
  {
    logTcp(*measured, "Measured start");
    const auto & reference = reference_targets_.at(step.pose);
    moveit::core::RobotState reference_state(*measured);
    for (const auto & joint : kArmJoints) {
      const double value = reference.at(joint);
      validatePosition(joint, value);
      reference_state.setVariablePosition(joint, value);
    }
    if (!reference_state.satisfiesBounds()) {
      throw std::runtime_error("Taught-reference IK seed violates MoveIt joint bounds");
    }
    const Eigen::Isometry3d reference_transform = logTcp(reference_state, "Taught reference FK");
    // Only this copy is seeded with taught arm joints; measured remains the
    // planning start and physical-displacement reference. Preserve its jaw state.
    moveit::core::RobotState solution_state(reference_state);
    Eigen::Isometry3d target_pose = reference_transform;
    target_pose.translation() = tcp_targets_.at(step.pose);
    // The pose contains valid reference orientation data, ignored by position-only KDL.
    kinematics::KinematicsQueryOptions options;
    options.return_approximate_solution = false;
    if (!solution_state.setFromIK(
        arm_group_, target_pose, target_link_, arm_group_->getDefaultIKTimeout(),
        moveit::core::GroupStateValidityCallbackFn(), options))
    {
      throw std::runtime_error("Position-only IK failed; no approximate solution or fallback");
    }
    const auto solution = positions(solution_state, kArmJoints);
    for (const auto & joint : kArmJoints) {
      validatePosition(joint, solution.at(joint));
    }
    if (!solution_state.satisfiesBounds(arm_group_)) {
      throw std::runtime_error("IK solution violates MoveIt arm bounds");
    }
    const Eigen::Isometry3d solution_transform = logTcp(solution_state, "IK solution FK");
    Eigen::Quaterniond reference_orientation(reference_transform.linear());
    Eigen::Quaterniond solution_orientation(solution_transform.linear());
    reference_orientation.normalize();
    solution_orientation.normalize();
    const double orientation_difference = reference_orientation.angularDistance(solution_orientation) *
      kRadiansToDegrees;
    if (!std::isfinite(orientation_difference)) {
      throw std::runtime_error("Non-finite taught-reference/IK orientation difference");
    }
    RCLCPP_INFO(node_->get_logger(),
      "Taught-reference to IK TCP quaternion angular difference=%.9f deg "
      "(diagnostic only; orientation unconstrained)", orientation_difference);
    validateReferencePosture(current, reference, solution, stage_limits_.at(step.pose));
    verifyTcp(solution_state, tcp_targets_.at(step.pose), "IK solution");
    // IK has no collision-validity callback. Mandatory MoveIt planning performs
    // collision checking against the existing configured planning scene.
    return solution;
  }

  void validateTrajectory(
    const MoveGroup::Plan & plan, const std::vector<std::string> & joints,
    const JointTarget & current, const JointTarget & target) const
  {
    const auto & trajectory = plan.trajectory_.joint_trajectory;
    if (trajectory.points.empty() || trajectory.joint_names.size() != joints.size() ||
      std::set<std::string>(trajectory.joint_names.begin(), trajectory.joint_names.end()) !=
      std::set<std::string>(joints.begin(), joints.end()) ||
      !plan.trajectory_.multi_dof_joint_trajectory.points.empty())
    {
      throw std::runtime_error("Empty or unexpected trajectory joint set");
    }
    std::int64_t previous_time = -1;
    for (const auto & point : trajectory.points) {
      if (point.positions.size() != trajectory.joint_names.size()) {
        throw std::runtime_error("Invalid trajectory position count");
      }
      if (point.time_from_start.sec < 0 || point.time_from_start.nanosec >= 1000000000u) {
        throw std::runtime_error("Invalid trajectory timestamp");
      }
      const std::int64_t timestamp = static_cast<std::int64_t>(point.time_from_start.sec) *
        1000000000LL + point.time_from_start.nanosec;
      if (timestamp <= previous_time) {
        throw std::runtime_error("Trajectory timestamps must strictly increase");
      }
      previous_time = timestamp;
      for (std::size_t index = 0; index < point.positions.size(); ++index) {
        const auto & joint = trajectory.joint_names[index];
        const double value = point.positions[index];
        validatePosition(joint, value);
        const double lower = std::min(current.at(joint), target.at(joint)) - trajectory_margin_;
        const double upper = std::max(current.at(joint), target.at(joint)) + trajectory_margin_;
        if (value < lower || value > upper) {
          throw std::runtime_error("Trajectory position outside stage envelope for " + joint);
        }
      }
      for (const auto * values : {&point.velocities, &point.accelerations, &point.effort}) {
        if ((!values->empty() && values->size() != trajectory.joint_names.size()) ||
          !std::all_of(values->begin(), values->end(),
            [](double value) {return std::isfinite(value);}))
        {
          throw std::runtime_error("Invalid/non-finite trajectory derivative or effort");
        }
      }
    }
    for (std::size_t index = 0; index < trajectory.joint_names.size(); ++index) {
      const auto & joint = trajectory.joint_names[index];
      if (std::abs(trajectory.points.front().positions[index] - current.at(joint)) >
        trajectory_start_tolerance_)
      {
        throw std::runtime_error("Trajectory first point disagrees with measured start for " + joint);
      }
      if (std::abs(trajectory.points.back().positions[index] - target.at(joint)) >
        trajectory_goal_tolerance_)
      {
        throw std::runtime_error("Trajectory final point disagrees with requested target for " + joint);
      }
    }
    RCLCPP_INFO(node_->get_logger(), "Validated trajectory: %zu joints, %zu points",
      trajectory.joint_names.size(), trajectory.points.size());
  }

  static void requireSuccess(const moveit::core::MoveItErrorCode & result, const char * operation)
  {
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      throw std::runtime_error(std::string(operation) + " failed: " +
        moveit::core::error_code_to_string(result) + " (" + std::to_string(result.val) + ")");
    }
  }

  void runStep(const Step & step)
  {
    const bool gripper_step = step.type == TargetType::Gripper;
    auto & group = gripper_step ? gripper_ : arm_;
    const auto & joints = gripper_step ? kGripperJoints : kArmJoints;
    const auto measured = measuredState();
    const auto current = positions(*measured, joints);
    JointTarget target;
    if (step.type == TargetType::Tcp) {
      target = solveTcp(step, measured, current);
    } else {
      target = joint_targets_.at(step.pose);
      validateTarget(current, target,
        gripper_step ? max_gripper_joint_delta_ : stage_limits_.at(step.pose));
    }
    group.setStartState(*measured);
    if (!group.setJointValueTarget(target)) {
      throw std::runtime_error("setJointValueTarget rejected " + std::string(step.pose));
    }
    MoveGroup::Plan plan;
    requireSuccess(group.plan(plan), "Plan");
    validateTrajectory(plan, joints, current, target);
    if (!execute_) {
      return;
    }
    // Reject drift during planning rather than silently replan. Check all six
    // joints so an unexpected jaw/arm change cannot invalidate the checked scene.
    const auto before_execute = measuredState();
    verifyJoints(*before_execute,
      positions(*measured, arm_.getRobotModel()->getVariableNames()),
      trajectory_start_tolerance_, "Pre-execute start");
    if (!rclcpp::ok()) {
      throw std::runtime_error("ROS shutdown before Execute");
    }
    // Execute precisely the validated plan and wait for its action result.
    requireSuccess(group.execute(plan), "Execute");
    const auto final_state = measuredState();
    if (step.type == TargetType::Tcp) {
      verifyTcp(*final_state, tcp_targets_.at(step.pose), "Final measured");
    } else {
      verifyJoints(*final_state, target,
        gripper_step ? gripper_joint_tolerance_ : home_joint_tolerance_, "Final measured");
      if (step.type == TargetType::Home) {
        const double error = (logTcp(*final_state, "HOME diagnostic").translation() -
          home_fk_reference_).norm();
        RCLCPP_INFO(node_->get_logger(), "HOME reference FK error=%.9f m (diagnostic only)", error);
      }
    }
  }

  rclcpp::Node::SharedPtr node_;
  bool execute_;
  std::string target_link_;
  std::string target_frame_;
  std::string stop_after_stage_;
  double state_timeout_;
  double max_fk_position_error_;
  double max_reference_joint_delta_;
  double trajectory_margin_;
  double home_joint_tolerance_;
  double gripper_joint_tolerance_;
  double trajectory_start_tolerance_;
  double trajectory_goal_tolerance_;
  double max_gripper_joint_delta_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  MoveGroup arm_;
  MoveGroup gripper_;
  planning_scene_monitor::CurrentStateMonitor state_monitor_;
  const moveit::core::JointModelGroup * arm_group_{nullptr};
  std::map<std::string, JointTarget> joint_targets_;
  std::map<std::string, JointTarget> reference_targets_;
  std::map<std::string, Eigen::Vector3d> tcp_targets_;
  JointTarget stage_limits_;
  Eigen::Vector3d home_fk_reference_;
  std::size_t stop_index_{kSequence.size() - 1};
  std::string stage_{"INITIALIZE"};
};
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const auto node = std::make_shared<rclcpp::Node>(
    "so101_tcp_pick_place",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  // Subscriptions and action results stay live while the main thread blocks.
  std::thread spin_thread([&executor]() {executor.spin();});
  int exit_code = EXIT_FAILURE;
  try {
    declareDefaults(node);
    TcpPickPlace task(node);
    exit_code = task.run() ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "[FAILED] stage=INITIALIZE: %s", error.what());
  }
  executor.cancel();
  spin_thread.join();
  rclcpp::shutdown();
  return exit_code;
}
