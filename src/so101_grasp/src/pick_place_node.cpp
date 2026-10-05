#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_monitor/current_state_monitor.h>
#include <moveit/utils/moveit_error_code.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>

#include <array>
#include <cmath>
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

struct Step
{
  const char * stage;
  const char * pose;
  bool gripper;
};

constexpr std::array<Step, 11> kSequence{{
  {"HOME", "home", false},
  {"OPEN_GRIPPER", "gripper_open", true},
  {"PRE_GRASP", "pre_grasp", false},
  {"APPROACH", "grasp", false},
  {"CLOSE_GRIPPER", "gripper_closed", true},
  {"LIFT", "lift", false},
  {"TRANSFER", "transfer", false},
  {"PLACE", "place", false},
  {"OPEN_GRIPPER", "gripper_open", true},
  {"RETREAT", "retreat", false},
  {"HOME", "home", false},
}};

double positiveParameter(const rclcpp::Node::SharedPtr & node, const std::string & name)
{
  const double value = node->get_parameter(name).as_double();
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(name + " must be finite and positive");
  }
  return value;
}

class PickPlace
{
public:
  explicit PickPlace(const rclcpp::Node::SharedPtr & node)
  : node_(node),
    state_timeout_(positiveParameter(node, "state_timeout")),
    max_delta_(positiveParameter(node, "max_joint_delta")),
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
      stage_ = "VALIDATE_POSES";
      prepare();
      for (std::size_t index = 0; index < kSequence.size(); ++index) {
        const auto & step = kSequence[index];
        stage_ = std::to_string(index + 1) + "/11 " + step.stage;
        RCLCPP_INFO(node_->get_logger(), "[START] %s pose=%s group=%s",
          stage_.c_str(), step.pose, step.gripper ? "gripper" : "arm");
        executeStep(step);
        RCLCPP_INFO(node_->get_logger(), "[DONE] %s", stage_.c_str());
      }
      RCLCPP_INFO(node_->get_logger(), "Pick & Place cycle complete (11/11 steps).");
      return true;
    } catch (const std::exception & error) {
      RCLCPP_ERROR(node_->get_logger(), "[FAILED] stage=%s: %s. State machine halted.",
        stage_.c_str(), error.what());
      // Never retry, open the gripper, or return home after a failure.
      if (rclcpp::ok()) {
        try {
          arm_.stop();
          gripper_.stop();
        } catch (const std::exception & stop_error) {
          RCLCPP_ERROR(node_->get_logger(), "Stop request failed: %s", stop_error.what());
        }
      }
      return false;
    }
  }

private:
  void validateGroup(MoveGroup & group, const std::vector<std::string> & expected)
  {
    const auto * model_group = group.getRobotModel()->getJointModelGroup(group.getName());
    if (!model_group) {
      throw std::runtime_error("Missing planning group " + group.getName());
    }
    const auto & actual = model_group->getVariableNames();
    if (std::set<std::string>(actual.begin(), actual.end()) !=
      std::set<std::string>(expected.begin(), expected.end()))
    {
      throw std::runtime_error("Unexpected joint names in group " + group.getName());
    }
  }

  moveit::core::RobotStatePtr measuredState()
  {
    if (!rclcpp::ok() ||
      !state_monitor_.waitForCurrentState(node_->now(), state_timeout_) ||
      !state_monitor_.haveCompleteState(rclcpp::Duration::from_seconds(1.0)))
    {
      throw std::runtime_error("current_state: missing/stale joint_states (all six required)");
    }
    auto state = state_monitor_.getCurrentState();
    for (const auto & joint : state->getVariableNames()) {
      validatePosition(joint, state->getVariablePosition(joint));
    }
    return state;
  }

  void validatePosition(const std::string & joint, double value) const
  {
    if (!std::isfinite(value)) {
      throw std::runtime_error("Non-finite position for " + joint);
    }
    const auto & bounds = arm_.getRobotModel()->getVariableBounds(joint);
    if (bounds.position_bounded_ &&
      (value < bounds.min_position_ || value > bounds.max_position_))
    {
      throw std::runtime_error("Joint limit violation for " + joint);
    }
  }

  void validateNearby(const std::string & joint, double value) const
  {
    validatePosition(joint, value);
    if (std::abs(value - baseline_.at(joint)) > max_delta_ + 1e-9) {
      throw std::runtime_error("Position outside startup neighborhood for " + joint);
    }
  }

  void prepare()
  {
    validateGroup(arm_, kArmJoints);
    validateGroup(gripper_, kGripperJoints);
    const double velocity = positiveParameter(node_, "velocity_scaling");
    const double acceleration = positiveParameter(node_, "acceleration_scaling");
    if (velocity > 0.1 || acceleration > 0.1 || max_delta_ > 0.1) {
      throw std::runtime_error("v1 requires scaling <= 0.1 and max_joint_delta <= 0.1 rad");
    }
    const double planning_time = positiveParameter(node_, "planning_time");
    for (auto * group : {&arm_, &gripper_}) {
      group->setMaxVelocityScalingFactor(velocity);
      group->setMaxAccelerationScalingFactor(acceleration);
      group->setPlanningTime(planning_time);
      group->setNumPlanningAttempts(1);
      group->allowReplanning(false);
    }

    const auto reference = node_->get_parameter("pose_reference").as_string();
    if (reference != "current_state" && reference != "absolute") {
      throw std::runtime_error("pose_reference must be current_state or absolute");
    }
    state_monitor_.startStateMonitor("joint_states");
    if (!state_monitor_.waitForCompleteState(state_timeout_)) {
      throw std::runtime_error("current_state: incomplete joint_states at startup");
    }
    const auto initial = measuredState();
    for (const auto & joint : initial->getVariableNames()) {
      baseline_.emplace(joint, initial->getVariablePosition(joint));
      RCLCPP_INFO(node_->get_logger(), "Startup joint %s = %.6f rad",
        joint.c_str(), baseline_.at(joint));
    }

    // Resolve every pose before the first movement. Never recapture a baseline
    // between steps: HOME at the end must equal HOME at the beginning.
    for (const auto & step : kSequence) {
      if (targets_.count(step.pose) != 0) {
        continue;
      }
      JointTarget values;
      node_->get_parameters("poses." + std::string(step.pose), values);
      const auto & joints = step.gripper ? kGripperJoints : kArmJoints;
      if (values.size() != joints.size()) {
        throw std::runtime_error("Wrong/missing joint keys for pose " + std::string(step.pose));
      }
      for (const auto & joint : joints) {
        if (values.count(joint) != 1) {
          throw std::runtime_error("Missing " + joint + " in pose " + step.pose);
        }
        if (reference == "current_state") {
          values.at(joint) += baseline_.at(joint);
        }
        validateNearby(joint, values.at(joint));
        RCLCPP_INFO(node_->get_logger(), "Resolved pose %s: %s = %.6f rad",
          step.pose, joint.c_str(), values.at(joint));
      }
      targets_.emplace(step.pose, std::move(values));
    }
    RCLCPP_WARN(node_->get_logger(),
      "Example poses require manual confirmation, including gripper open/close direction.");
  }

  void validateTrajectory(const MoveGroup::Plan & plan, const std::vector<std::string> & joints)
  {
    const auto & trajectory = plan.trajectory_.joint_trajectory;
    if (trajectory.points.empty() || trajectory.joint_names.size() != joints.size() ||
      std::set<std::string>(trajectory.joint_names.begin(), trajectory.joint_names.end()) !=
      std::set<std::string>(joints.begin(), joints.end()) ||
      !plan.trajectory_.multi_dof_joint_trajectory.points.empty())
    {
      throw std::runtime_error("plan_validation: empty or unexpected trajectory");
    }
    // A small target can still produce a large planner detour. Reject that
    // solution before executing, rather than assuming all paths stay nearby.
    for (const auto & point : trajectory.points) {
      if (point.positions.size() != trajectory.joint_names.size()) {
        throw std::runtime_error("plan_validation: invalid trajectory positions");
      }
      for (std::size_t index = 0; index < point.positions.size(); ++index) {
        validateNearby(trajectory.joint_names[index], point.positions[index]);
      }
    }
  }

  void executeStep(const Step & step)
  {
    auto & group = step.gripper ? gripper_ : arm_;
    const auto state = measuredState();
    for (const auto & joint : state->getVariableNames()) {
      validateNearby(joint, state->getVariablePosition(joint));
    }
    group.setStartState(*state);
    if (!group.setJointValueTarget(targets_.at(step.pose))) {
      throw std::runtime_error("setJointValueTarget rejected pose " + std::string(step.pose));
    }
    MoveGroup::Plan plan;
    const auto plan_result = group.plan(plan);
    if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
      throw std::runtime_error("Plan failed: " + moveit::core::error_code_to_string(plan_result) +
        " (" + std::to_string(plan_result.val) + ")");
    }
    validateTrajectory(plan, step.gripper ? kGripperJoints : kArmJoints);
    if (!rclcpp::ok()) {
      throw std::runtime_error("ROS shutdown before Execute");
    }
    // execute() waits for the execution action result; no sleep or second plan.
    const auto execute_result = group.execute(plan);
    if (execute_result != moveit::core::MoveItErrorCode::SUCCESS) {
      throw std::runtime_error("Execute failed: " +
        moveit::core::error_code_to_string(execute_result) +
        " (" + std::to_string(execute_result.val) + ")");
    }
  }

  rclcpp::Node::SharedPtr node_;
  double state_timeout_;
  double max_delta_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  MoveGroup arm_;
  MoveGroup gripper_;
  planning_scene_monitor::CurrentStateMonitor state_monitor_;
  JointTarget baseline_;
  std::map<std::string, JointTarget> targets_;
  std::string stage_{"INITIALIZE"};
};
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const auto node = std::make_shared<rclcpp::Node>(
    "so101_pick_place", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  // Keep subscriptions and action results alive while the main thread blocks
  // in plan()/execute(). Moving the sequence into a timer would block callbacks.
  std::thread spin_thread([&executor]() {executor.spin();});
  int exit_code = EXIT_FAILURE;
  try {
    PickPlace task(node);
    exit_code = task.run() ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "[FAILED] stage=INITIALIZE: %s", error.what());
  }
  executor.cancel();
  spin_thread.join();
  rclcpp::shutdown();
  return exit_code;
}
