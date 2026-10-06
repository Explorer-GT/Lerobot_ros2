#include <moveit/kinematics_base/kinematics_base.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_monitor/current_state_monitor.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/utils/moveit_error_code.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
using MoveGroup = moveit::planning_interface::MoveGroupInterface;
using JointTarget = std::map<std::string, double>;
const std::vector<std::string> kArmJoints{
  "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"};
// Match the existing pick/place trajectory envelope: at most 0.15 rad of
// overshoot beyond each measured-start to IK-solution interval.
constexpr double kTrajectoryMargin = 0.15;
constexpr double kMaxStateAge = 1.0;

double positiveParameter(const rclcpp::Node::SharedPtr & node, const std::string & name)
{
  const double value = node->get_parameter(name).as_double();
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(name + " must be finite and positive");
  }
  return value;
}

class XyzIkTest
{
public:
  explicit XyzIkTest(const rclcpp::Node::SharedPtr & node)
  : node_(node),
    target_link_(node->get_parameter("target_link").as_string()),
    execute_(node->get_parameter("execute").as_bool()),
    state_timeout_(positiveParameter(node, "state_timeout")),
    max_ik_joint_delta_(positiveParameter(node, "max_ik_joint_delta")),
    max_fk_position_error_(positiveParameter(node, "max_fk_position_error")),
    tf_buffer_(std::make_shared<tf2_ros::Buffer>(node->get_clock())),
    arm_(node, "arm", tf_buffer_, rclcpp::Duration::from_seconds(10.0)),
    state_monitor_(node, arm_.getRobotModel(), tf_buffer_, false)
  {
    offset_ = Eigen::Vector3d(
      node->get_parameter("dx").as_double(),
      node->get_parameter("dy").as_double(),
      node->get_parameter("dz").as_double());
    if (!offset_.allFinite()) {
      throw std::runtime_error("dx, dy and dz must be finite");
    }
  }

  bool run()
  {
    try {
      stage_ = "PREPARE";
      prepare();

      stage_ = "CURRENT_STATE";
      state_monitor_.startStateMonitor("/joint_states");
      const auto measured = measuredState();
      const auto current = armPositions(*measured);

      // getGlobalLinkTransform() and setFromIK() both use the RobotModel frame.
      // Keep a valid seed orientation in the pose; KDL position-only IK ignores
      // that orientation. Only the gripper_link origin is tested here.
      Eigen::Isometry3d target_pose = measured->getGlobalLinkTransform(target_link_);
      if (!target_pose.matrix().allFinite()) {
        throw std::runtime_error("Non-finite current end-link transform");
      }
      logPosition("current", target_pose.translation());
      target_pose.translation() += offset_;
      const Eigen::Vector3d target_position = target_pose.translation();
      if (!target_position.allFinite()) {
        throw std::runtime_error("Non-finite target position");
      }
      logPosition("target", target_position);

      stage_ = "IK";
      moveit::core::RobotState solution_state(*measured);
      kinematics::KinematicsQueryOptions options;
      options.return_approximate_solution = false;
      if (!solution_state.setFromIK(
          arm_group_, target_pose, target_link_, arm_group_->getDefaultIKTimeout(),
          moveit::core::GroupStateValidityCallbackFn(), options))
      {
        throw std::runtime_error("Position-only IK failed; no approximate solution or fallback");
      }

      stage_ = "IK_VALIDATION";
      const auto solution = armPositions(solution_state);
      for (const auto & joint : kArmJoints) {
        const double value = solution.at(joint);
        const double delta = std::abs(value - current.at(joint));
        RCLCPP_INFO(node_->get_logger(),
          "IK joint %s: current=%.6f solution=%.6f delta=%.6f rad",
          joint.c_str(), current.at(joint), value, delta);
      }
      for (const auto & joint : kArmJoints) {
        const double delta = std::abs(solution.at(joint) - current.at(joint));
        if (!std::isfinite(delta) || delta > max_ik_joint_delta_) {
          throw std::runtime_error("IK joint delta exceeds max_ik_joint_delta for " + joint);
        }
      }
      for (const auto & joint : kArmJoints) {
        validatePosition(joint, solution.at(joint));
      }
      if (!solution_state.satisfiesBounds(arm_group_)) {
        throw std::runtime_error("IK solution violates MoveIt joint bounds");
      }

      stage_ = "IK_FK_VALIDATION";
      checkPositionError(solution_state, target_position, "IK FK", "position error");

      stage_ = "PLAN";
      // Preserve the original measured state; IK modified only its copy.
      arm_.setStartState(*measured);
      if (!arm_.setJointValueTarget(solution)) {
        throw std::runtime_error("setJointValueTarget rejected the IK solution");
      }
      MoveGroup::Plan plan;
      const auto plan_result = arm_.plan(plan);
      requireSuccess(plan_result, "Plan");

      stage_ = "PLAN_VALIDATION";
      validateTrajectory(plan, current, solution);
      if (!execute_) {
        RCLCPP_INFO(node_->get_logger(), "PLAN ONLY SUCCESS");
        return true;
      }

      stage_ = "EXECUTE";
      if (!rclcpp::ok()) {
        throw std::runtime_error("ROS shutdown before Execute");
      }
      // Execute precisely the checked plan, without replanning or sleeps.
      requireSuccess(arm_.execute(plan), "Execute");

      stage_ = "FINAL_STATE";
      // Require updates for all six joints after execute() has returned.
      const auto final_state = measuredState();
      stage_ = "FINAL_FK_VALIDATION";
      checkPositionError(*final_state, target_position, "final", "final position error");
      RCLCPP_INFO(node_->get_logger(), "EXECUTE SUCCESS");
      return true;
    } catch (const std::exception & error) {
      RCLCPP_ERROR(node_->get_logger(), "[FAILED] stage=%s: %s. Test halted.",
        stage_.c_str(), error.what());
      return false;
    }
  }

private:
  void prepare()
  {
    if (target_link_ != "gripper_link") {
      throw std::runtime_error("This first-version test requires target_link=gripper_link");
    }
    arm_group_ = arm_.getRobotModel()->getJointModelGroup("arm");
    if (!arm_group_ || !arm_group_->isChain()) {
      throw std::runtime_error("Missing or non-chain arm planning group");
    }
    const auto & variables = arm_group_->getVariableNames();
    if (variables.size() != kArmJoints.size() ||
      std::set<std::string>(variables.begin(), variables.end()) !=
      std::set<std::string>(kArmJoints.begin(), kArmJoints.end()))
    {
      throw std::runtime_error("arm must contain exactly the five expected joints");
    }
    const std::string prefix = "robot_description_kinematics.arm.";
    if (node_->get_parameter(prefix + "kinematics_solver").as_string() !=
      "kdl_kinematics_plugin/KDLKinematicsPlugin" ||
      !node_->get_parameter(prefix + "position_only_ik").as_bool())
    {
      throw std::runtime_error("arm requires KDL with position_only_ik=true");
    }
    const auto solver = arm_group_->getSolverInstance();
    if (!solver || solver->getTipFrames().size() != 1 ||
      solver->getTipFrames().front() != target_link_)
    {
      throw std::runtime_error("Missing IK solver or solver tip differs from gripper_link");
    }
    const double ik_timeout = arm_group_->getDefaultIKTimeout();
    if (!std::isfinite(ik_timeout) || ik_timeout <= 0.0) {
      throw std::runtime_error("Invalid arm IK timeout");
    }
    const double velocity = positiveParameter(node_, "velocity_scaling");
    const double acceleration = positiveParameter(node_, "acceleration_scaling");
    if (velocity > 0.1 || acceleration > 0.1) {
      throw std::runtime_error("Fake-hardware tests require velocity/acceleration scaling <= 0.1");
    }
    arm_.setMaxVelocityScalingFactor(velocity);
    arm_.setMaxAccelerationScalingFactor(acceleration);
    arm_.setPlanningTime(5.0);
    arm_.setNumPlanningAttempts(1);
    arm_.allowReplanning(false);
    RCLCPP_INFO(node_->get_logger(),
      "Test link=%s (link origin, not a calibrated grasp center), frame=%s, "
      "IK base=%s, IK timeout=%.3f s, execute=%s",
      target_link_.c_str(), arm_.getRobotModel()->getModelFrame().c_str(),
      solver->getBaseFrame().c_str(), ik_timeout, execute_ ? "true" : "false");
    RCLCPP_INFO(node_->get_logger(),
      "Limits: max_ik_joint_delta=%.6f rad, max_fk_position_error=%.6f m, "
      "trajectory_margin=%.6f rad",
      max_ik_joint_delta_, max_fk_position_error_, kTrajectoryMargin);
  }

  moveit::core::RobotStatePtr measuredState()
  {
    const auto requested_time = node_->now();
    auto next_update = requested_time;
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration<double>(state_timeout_);
    // waitForCurrentState uses the monitor's condition variable. Requiring
    // completeness separately also supports publishers that split the joints
    // across messages; a single recently updated joint is never enough.
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

  JointTarget armPositions(const moveit::core::RobotState & state) const
  {
    JointTarget positions;
    for (const auto & joint : kArmJoints) {
      positions.emplace(joint, state.getVariablePosition(joint));
    }
    return positions;
  }

  void logPosition(const char * label, const Eigen::Vector3d & position) const
  {
    RCLCPP_INFO(node_->get_logger(), "%s x=%.6f y=%.6f z=%.6f m",
      label, position.x(), position.y(), position.z());
  }

  void checkPositionError(
    moveit::core::RobotState & state, const Eigen::Vector3d & target,
    const char * position_label, const char * error_label) const
  {
    state.update();
    const Eigen::Vector3d actual = state.getGlobalLinkTransform(target_link_).translation();
    if (!actual.allFinite()) {
      throw std::runtime_error("Non-finite FK position");
    }
    const double error = (actual - target).norm();
    logPosition(position_label, actual);
    logPosition("target", target);
    RCLCPP_INFO(node_->get_logger(), "%s=%.9f m (maximum=%.6f m)",
      error_label, error, max_fk_position_error_);
    if (!std::isfinite(error) || error > max_fk_position_error_) {
      throw std::runtime_error(std::string(error_label) + " exceeds max_fk_position_error");
    }
  }

  void validateTrajectory(
    const MoveGroup::Plan & plan, const JointTarget & current, const JointTarget & solution) const
  {
    const auto & trajectory = plan.trajectory_.joint_trajectory;
    if (trajectory.points.empty() || trajectory.joint_names.size() != kArmJoints.size() ||
      std::set<std::string>(trajectory.joint_names.begin(), trajectory.joint_names.end()) !=
      std::set<std::string>(kArmJoints.begin(), kArmJoints.end()) ||
      !plan.trajectory_.multi_dof_joint_trajectory.points.empty())
    {
      throw std::runtime_error("Empty or unexpected arm trajectory");
    }
    for (const auto & point : trajectory.points) {
      if (point.positions.size() != trajectory.joint_names.size()) {
        throw std::runtime_error("Invalid trajectory position count");
      }
      for (std::size_t index = 0; index < point.positions.size(); ++index) {
        const auto & joint = trajectory.joint_names[index];
        const double value = point.positions[index];
        validatePosition(joint, value);
        const double lower = std::min(current.at(joint), solution.at(joint)) - kTrajectoryMargin;
        const double upper = std::max(current.at(joint), solution.at(joint)) + kTrajectoryMargin;
        if (value < lower || value > upper) {
          throw std::runtime_error("Trajectory detour for " + joint + " outside [" +
            std::to_string(lower) + ", " + std::to_string(upper) + "]");
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
    RCLCPP_INFO(node_->get_logger(), "Validated arm trajectory: %zu points",
      trajectory.points.size());
  }

  static void requireSuccess(const moveit::core::MoveItErrorCode & result, const char * operation)
  {
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      throw std::runtime_error(std::string(operation) + " failed: " +
        moveit::core::error_code_to_string(result) + " (" + std::to_string(result.val) + ")");
    }
  }

  rclcpp::Node::SharedPtr node_;
  std::string target_link_;
  bool execute_;
  double state_timeout_;
  double max_ik_joint_delta_;
  double max_fk_position_error_;
  Eigen::Vector3d offset_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  MoveGroup arm_;
  planning_scene_monitor::CurrentStateMonitor state_monitor_;
  const moveit::core::JointModelGroup * arm_group_{nullptr};
  std::string stage_{"INITIALIZE"};
};
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const auto node = std::make_shared<rclcpp::Node>(
    "so101_xyz_ik_test", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  // Action results and state subscriptions remain live while run() blocks.
  std::thread spin_thread([&executor]() {executor.spin();});
  int exit_code = EXIT_FAILURE;
  try {
    XyzIkTest test(node);
    exit_code = test.run() ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "[FAILED] stage=INITIALIZE: %s", error.what());
  }
  executor.cancel();
  spin_thread.join();
  rclcpp::shutdown();
  return exit_code;
}
