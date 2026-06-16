#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# ur_dual_moveit_py.py
#
# MoveItPy wrapper for the dual UR5 + SoftHand system.
#
# Scope (only the logical right arm is driven):
#   - MoveIt group "Right_arm" (physically the ur_dual_I_* arm, holds the SoftHand)
#   - OMPL planning to named poses and to Cartesian poses (best-of-N, cable-aware)
#   - Real straight-line Cartesian planning via /compute_cartesian_path
#   - OctoMap relay: mirrors move_group's OctoMap into this node's planning scene
# -----------------------------------------------------------------------------

import time
import rclpy
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import GetCartesianPath
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from moveit.planning import MoveItPy, PlanningComponent
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.robot_state import RobotState, robotStateToRobotStateMsg
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState as RobotStateMsg
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
)

# Degrees per radian, used only for human-readable joint-motion logs.
RAD_TO_DEG = 57.2958


class UrDualMoveItPy(Node):
    """Base wrapper to drive the dual UR5 from MoveItPy."""

    def __init__(self, name: str = "ur_dual_commander"):
        super().__init__(name)

        self.get_logger().info("Initializing UrDualMoveItPy...")

        # MoveItPy must use the same node name (see header note).
        self.robot = MoveItPy(node_name=name)

        self.robot_model = self.robot.get_robot_model()
        self.planning_monitor = self.robot.get_planning_scene_monitor()

        # Trajectory execution settings (allow some duration slack and a small
        # start tolerance so execution is not rejected by tiny state mismatches).
        try:
            trajectory_execution = self.robot.get_trajectory_execution_manager()
            trajectory_execution.enable_execution_duration_monitoring(True)
            trajectory_execution.set_allowed_execution_duration_scaling(1.2)
            trajectory_execution.set_allowed_start_tolerance(0.05)
            self.get_logger().info("TrajectoryExecutionManager configured.")
        except Exception as exc:
            self.get_logger().warn(
                f"Could not configure TrajectoryExecutionManager: {exc}"
            )

        # Only the right arm is used. "Right_arm" maps physically to the ur_dual_I_* arm.
        self.groups: Dict[str, PlanningComponent] = {
            "right": self.robot.get_planning_component("Right_arm"),
        }

        # Link used as the TCP for Cartesian goals. tool0 is inside the planning group.
        self.pose_links: Dict[str, str] = {
            "right": "ur_dual_I_tool0",
        }

        # Publisher for planning-scene diffs. Used by the commander's attach/detach
        # services to push AttachedCollisionObjects to move_group on /planning_scene.
        self.planning_scene_pub = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            10,
        )

        # --- OctoMap relay ---
        # This node's PlanningSceneMonitor has no 3D sensor of its own, so it never
        # builds an OctoMap. move_group publishes its full scene (with the
        # category-filtered OctoMap) on /monitored_planning_scene; we subscribe and
        # re-inject only the OctoMap into the local scene so planning here respects it.
        self._octomap_synced_once = False
        self._octomap_sync_sub = self.create_subscription(
            PlanningScene,
            "/monitored_planning_scene",
            self._on_move_group_scene,
            10,
        )

        # Dedicated callback group for internal service clients, so a service of ours
        # can call another ROS 2 service without deadlocking.
        self.cb_internal_client = MutuallyExclusiveCallbackGroup()

        self.cartesian_plan_client = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
            callback_group=self.cb_internal_client,
        )

        self.get_logger().info("UrDualMoveItPy ready.")

    # Internal helpers

    def _get_arm(self, arm: str) -> PlanningComponent:
        if arm not in self.groups:
            raise ValueError(f"Invalid arm: {arm}. Options: {list(self.groups)}")
        return self.groups[arm]

    def _get_pose_link(self, arm: str, pose_link: Optional[str] = None) -> str:
        return pose_link if pose_link is not None else self.pose_links[arm]


    # General planning / execution

    def plan(self, planning_component: PlanningComponent):
        """Run planning_component.plan() and return the result."""
        self.get_logger().info("Planning trajectory...")
        return planning_component.plan()

    def execute_trajectory(
        self,
        trajectory: RobotTrajectory,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
        sleep_time: float = 0.0,
    ):
        """Execute a RobotTrajectory through the configured controllers."""

        self.get_logger().warn(
            "Executing trajectory. Stay alert and keep the e-stop ready."
        )

        # Re-time the trajectory (TOTG). If it fails, execute it as-is.
        try:
            trajectory.apply_totg_time_parameterization(
                velocity_scaling,
                acceleration_scaling,
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Could not apply TOTG manually; executing trajectory as-is: {exc}"
            )

        result = self.robot.execute(trajectory, controllers=[])
        time.sleep(sleep_time)

        self.get_logger().info(f"Execution result: {result}")
        return result


    # Cable-protection checks
    #
    # These do NOT replace MoveIt collision checking. They are an extra guard for
    # the SoftHand cable: OMPL can return a valid but needlessly twisted path, so
    # we reject/score trajectories by how much the joints rotate.
    def _trajectory_cable_motion_ok(
        self,
        trajectory: RobotTrajectory,
        max_delta_by_joint: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Reject the trajectory if any arm joint sweeps more than its limit (rad)."""

        if max_delta_by_joint is None:
            # Per-joint hard limits (rad). The wrist joints are tighter because the
            # SoftHand cable suffers most there.
            max_delta_by_joint = {
                "ur_dual_I_shoulder_pan_joint": 3.80,
                "ur_dual_I_shoulder_lift_joint": 3.40,
                "ur_dual_I_elbow_joint": 3.40,
                "ur_dual_I_wrist_1_joint": 1.70,
                "ur_dual_I_wrist_2_joint": 1.55,
                "ur_dual_I_wrist_3_joint": 0.75,
            }

        traj_msg = trajectory.get_robot_trajectory_msg()
        joint_names = list(traj_msg.joint_trajectory.joint_names)

        if not traj_msg.joint_trajectory.points:
            self.get_logger().error("Empty trajectory. Rejected for safety.")
            return False

        ok = True
        self.get_logger().info("Cable/joint sweep check:")

        for joint_name, max_delta_rad in max_delta_by_joint.items():
            if joint_name not in joint_names:
                self.get_logger().warn(f"  {joint_name}: not in trajectory.")
                continue

            idx = joint_names.index(joint_name)
            positions = [
                point.positions[idx]
                for point in traj_msg.joint_trajectory.points
                if len(point.positions) > idx
            ]
            if not positions:
                self.get_logger().warn(f"  {joint_name}: no positions in trajectory.")
                continue

            # Sweep = max - min over the whole trajectory for this joint.
            delta = max(positions) - min(positions)
            self.get_logger().info(
                f"  {joint_name}: delta={delta:.3f} rad "
                f"({delta * RAD_TO_DEG:.1f} deg), limit={max_delta_rad:.3f} rad "
                f"({max_delta_rad * RAD_TO_DEG:.1f} deg)"
            )

            if abs(delta) > max_delta_rad:
                ok = False
                self.get_logger().error(f"  REJECTED: {joint_name} rotates too much.")

        if not ok:
            self.get_logger().error("Trajectory rejected by cable protection.")

        return ok

    def _trajectory_cable_cost(self, trajectory: RobotTrajectory) -> tuple[float, dict]:
        """Weighted joint-sweep cost used to pick the gentlest of several valid plans.

        Lower cost = friendlier to the cable. This does NOT decide collisions;
        MoveIt has already validated them. It only ranks valid trajectories.
        """

        traj_msg = trajectory.get_robot_trajectory_msg()
        joint_names = list(traj_msg.joint_trajectory.joint_names)

        # Equal weights for now; kept as a dict so the wrist can be weighted higher later.
        weights = {
            "ur_dual_I_shoulder_pan_joint": 0.001,
            "ur_dual_I_shoulder_lift_joint": 0.001,
            "ur_dual_I_elbow_joint": 0.001,
            "ur_dual_I_wrist_1_joint": 0.001,
            "ur_dual_I_wrist_2_joint": 0.001,
            "ur_dual_I_wrist_3_joint": 0.001,
        }

        deltas = {}
        cost = 0.0

        for joint_name, weight in weights.items():
            if joint_name not in joint_names:
                continue
            idx = joint_names.index(joint_name)
            positions = [
                point.positions[idx]
                for point in traj_msg.joint_trajectory.points
                if len(point.positions) > idx
            ]
            if not positions:
                continue
            delta = max(positions) - min(positions)
            deltas[joint_name] = delta
            cost += weight * abs(delta)

        return cost, deltas

    def _log_cable_deltas(self, deltas: dict, prefix: str = ""):
        """Pretty-print per-joint sweeps."""
        if prefix:
            self.get_logger().info(prefix)
        for joint_name, delta in deltas.items():
            self.get_logger().info(
                f"  {joint_name}: delta={delta:.3f} rad ({delta * RAD_TO_DEG:.1f} deg)"
            )


    # OMPL planning to a Cartesian pose (best-of-N, cable-aware)


    def arm_plan_to_pose_raw(
        self,
        arm: str,
        pose: PoseStamped,
        pose_link: Optional[str] = None,
        orientation_tolerance: Optional[tuple] = None,
        position_tolerance: float = 0.005,
    ):
        """Plan to a pose and return the raw plan (no cable filter applied).

        Used to generate candidates that a caller then ranks.

        If orientation_tolerance is None, plan to the exact pose with MoveIt's
        default tolerance. If it is (tol_roll, tol_pitch, tol_yaw) in radians,
        build a goal from a tight PositionConstraint (sphere of radius
        position_tolerance) plus a per-axis OrientationConstraint. Relaxing the
        orientation (especially yaw) lets OMPL explore many more IK solutions.
        """

        selected_arm = self._get_arm(arm)
        link = self._get_pose_link(arm, pose_link)

        selected_arm.set_start_state_to_current_state()

        if orientation_tolerance is None:
            # Exact pose goal.
            selected_arm.set_goal_state(pose_stamped_msg=pose, pose_link=link)
            mode_msg = "exact pose"
        else:
            # Goal with constraints (tight position + tolerant orientation).
            constraints = self._build_pose_constraints(
                pose=pose,
                link=link,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
            selected_arm.set_goal_state(motion_plan_constraints=[constraints])
            tol_r, tol_p, tol_y = orientation_tolerance
            mode_msg = (
                f"constraints (pos={position_tolerance*1000:.0f}mm, "
                f"ori=({tol_r:.2f},{tol_p:.2f},{tol_y:.2f}) rad)"
            )

        self.get_logger().info(
            f"Candidate plan to pose using pose_link='{link}' "
            f"frame='{pose.header.frame_id}' [{mode_msg}]"
        )

        plan_result = selected_arm.plan()
        if not plan_result:
            return None
        return plan_result

    def _build_pose_constraints(
        self,
        pose: PoseStamped,
        link: str,
        position_tolerance: float,
        orientation_tolerance: tuple,
    ) -> Constraints:
        """Build a Constraints goal: a position sphere + per-axis orientation tolerance.

        This is what OMPL receives when we pass motion_plan_constraints instead of a
        plain pose goal.
        """
        constraints = Constraints()
        constraints.name = "pregrasp_target_with_tol"

        # PositionConstraint: a sphere of radius position_tolerance around the target.
        pos_c = PositionConstraint()
        pos_c.header = pose.header
        pos_c.link_name = link
        pos_c.target_point_offset.x = 0.0
        pos_c.target_point_offset.y = 0.0
        pos_c.target_point_offset.z = 0.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [position_tolerance]
        pos_c.constraint_region.primitives.append(sphere)

        from geometry_msgs.msg import Pose as PoseMsg
        region_pose = PoseMsg()
        region_pose.position = pose.pose.position
        region_pose.orientation.w = 1.0
        pos_c.constraint_region.primitive_poses.append(region_pose)
        pos_c.weight = 1.0
        constraints.position_constraints.append(pos_c)

        # OrientationConstraint: reference orientation with per-axis (XYZ-Euler) tolerance.
        tol_roll, tol_pitch, tol_yaw = orientation_tolerance
        ori_c = OrientationConstraint()
        ori_c.header = pose.header
        ori_c.link_name = link
        ori_c.orientation = pose.pose.orientation
        ori_c.absolute_x_axis_tolerance = float(tol_roll)
        ori_c.absolute_y_axis_tolerance = float(tol_pitch)
        ori_c.absolute_z_axis_tolerance = float(tol_yaw)
        ori_c.parameterization = OrientationConstraint.XYZ_EULER_ANGLES
        ori_c.weight = 1.0
        constraints.orientation_constraints.append(ori_c)

        return constraints

    def arm_go_to_pose_best_of_n(
        self,
        arm: str,
        pose: PoseStamped,
        pose_link: Optional[str] = None,
        attempts: int = 12,
        execute: bool = False,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
        orientation_tolerance: Optional[tuple] = None,
        position_tolerance: float = 0.005,
    ):
        """Generate several OMPL plans and keep the one with the lowest cable cost.

        Returns (success, status, plan). On success the chosen plan still has to pass
        the hard cable-sweep limits before it is accepted.
        """

        best_plan = None
        best_cost = None
        best_deltas = None

        self.get_logger().info(
            f"Starting best-of-{attempts} pre-grasp with cable protection."
        )

        for i in range(attempts):
            self.get_logger().info(f"OMPL candidate {i + 1}/{attempts}")

            plan_result = self.arm_plan_to_pose_raw(
                arm=arm,
                pose=pose,
                pose_link=pose_link,
                orientation_tolerance=orientation_tolerance,
                position_tolerance=position_tolerance,
            )
            if plan_result is None:
                self.get_logger().warn(f"  Attempt {i + 1}: PLAN_FAILED")
                continue

            cost, deltas = self._trajectory_cable_cost(plan_result.trajectory)
            self.get_logger().info(f"  Attempt {i + 1}: cable cost={cost:.3f}")
            self._log_cable_deltas(deltas)

            if best_plan is None or cost < best_cost:
                best_plan = plan_result
                best_cost = cost
                best_deltas = deltas

        if best_plan is None:
            self.get_logger().error("No OMPL attempt produced a valid trajectory.")
            return False, "PLAN_FAILED", None

        self.get_logger().info(f"Best plan selected with cable cost={best_cost:.3f}")
        self._log_cable_deltas(best_deltas, prefix="Best-plan deltas:")

        # Apply the hard cable limits to the chosen plan.
        if not self._trajectory_cable_motion_ok(best_plan.trajectory):
            self.get_logger().error("Best plan still too risky for the cable.")
            return False, "CABLE_MOTION_TOO_LARGE", best_plan

        self.get_logger().info("BEST-OF-N PLAN OK")

        if execute:
            exec_result = self.execute_trajectory(
                best_plan.trajectory,
                velocity_scaling=velocity_scaling,
                acceleration_scaling=acceleration_scaling,
            )
            return True, str(exec_result), best_plan

        return True, "BEST_OF_N_PLAN_SUCCEEDED", best_plan

    def plan_and_maybe_execute(
        self,
        planning_component: PlanningComponent,
        execute: bool = False,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
    ):
        """Plan and, if execute=True, run it. Applies the cable-sweep check first."""

        plan_result = self.plan(planning_component)

        if not plan_result:
            self.get_logger().error(
                "PLAN FAILED. Collision, unreachable goal, or timeout."
            )
            return False, "PLAN_FAILED", None

        self.get_logger().info("PLAN OK. Trajectory validated by the MoveIt pipeline.")

        if not self._trajectory_cable_motion_ok(plan_result.trajectory):
            self.get_logger().error(
                "PLAN REJECTED: MoveIt found a trajectory but the joint motion is "
                "risky for the cable."
            )
            return False, "CABLE_MOTION_TOO_LARGE", plan_result

        if execute:
            exec_result = self.execute_trajectory(
                plan_result.trajectory,
                velocity_scaling=velocity_scaling,
                acceleration_scaling=acceleration_scaling,
            )
            return True, str(exec_result), plan_result

        return True, "PLAN_SUCCEEDED", plan_result


    # Arm moves


    def arm_go_to_named_pose(
        self,
        arm: str,
        pose_name: str,
        execute: bool = False,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
    ):
        """Plan/execute to an SRDF named pose (e.g. Ready_Right, Place_Normal_Right)."""

        selected_arm = self._get_arm(arm)
        selected_arm.set_start_state_to_current_state()
        selected_arm.set_goal_state(configuration_name=pose_name)

        self.get_logger().info(
            f"Arm '{arm}' -> named pose '{pose_name}' | execute={execute}"
        )

        return self.plan_and_maybe_execute(
            selected_arm,
            execute=execute,
            velocity_scaling=velocity_scaling,
            acceleration_scaling=acceleration_scaling,
        )


    # OctoMap relay


    def _on_move_group_scene(self, msg: PlanningScene):
        """Relay ONLY the OctoMap from move_group into the local planning scene as a diff."""
        octo = msg.world.octomap.octomap

        # Skip messages without an OctoMap (including the empty scenes this node itself
        # publishes on the same topic). This prevents wiping the map we already hold.
        if octo.resolution <= 0.0 or len(octo.data) == 0:
            return

        diff = PlanningScene()
        diff.is_diff = True
        diff.world.octomap = msg.world.octomap  # only the octomap, nothing else

        try:
            self.planning_monitor.new_planning_scene_message(diff)
        except Exception as exc:
            self.get_logger().warn(f"Could not apply move_group OctoMap: {exc}")
            return

        if not self._octomap_synced_once:
            self._octomap_synced_once = True
            self.get_logger().info(
                f"move_group OctoMap synced into local PSM "
                f"(resolution={octo.resolution:.3f}, bytes={len(octo.data)})."
            )

    def clear_local_octomap(self) -> tuple[bool, str]:
        """Clear the OctoMap held by THIS MoveItPy node's PlanningSceneMonitor.

        /clear_octomap clears move_group's map; this project plans from the
        commander's local PSM, so we must clear that one. The Python binding name
        varies, so we probe a couple of candidates on the monitor and on the scene.
        """

        # Option 1: method exposed on the PlanningSceneMonitor.
        for method_name in ("clear_octomap", "clearOctomap"):
            if hasattr(self.planning_monitor, method_name):
                try:
                    getattr(self.planning_monitor, method_name)()
                    self.get_logger().info(
                        f"Local OctoMap cleared via planning_monitor.{method_name}()."
                    )
                    return True, f"Local OctoMap cleared with {method_name}()."
                except Exception as exc:
                    self.get_logger().warn(f"planning_monitor.{method_name}() failed: {exc}")

        # Option 2: some bindings expose it on the scene object instead.
        try:
            with self.planning_monitor.read_write() as scene:
                for method_name in ("clear_octomap", "clearOctomap"):
                    if hasattr(scene, method_name):
                        try:
                            getattr(scene, method_name)()
                            scene.current_state.update()
                            self.get_logger().info(
                                f"Local OctoMap cleared via scene.{method_name}()."
                            )
                            return True, f"Local OctoMap cleared with scene.{method_name}()."
                        except Exception as exc:
                            self.get_logger().warn(f"scene.{method_name}() failed: {exc}")
        except Exception as exc:
            self.get_logger().warn(
                f"Could not open planning_monitor.read_write() to clear OctoMap: {exc}"
            )

        msg = (
            "No exposed Python method found to clear the local OctoMap. "
            "As a fallback, restart ur_dual_command.launch.py to reset the map."
        )
        self.get_logger().error(msg)
        return False, msg


    # Real Cartesian planning (/compute_cartesian_path)


    def _trajectory_joint_delta_ok(
        self,
        trajectory: RobotTrajectory,
        joint_name: str = "ur_dual_I_wrist_3_joint",
        max_delta_rad: float = 0.35,
    ) -> bool:
        """Reject a trajectory if a single joint sweeps too much (cable guard).

        Default 0.35 rad (~20 deg) on wrist_3.
        """

        traj_msg = trajectory.get_robot_trajectory_msg()
        joint_names = list(traj_msg.joint_trajectory.joint_names)

        if joint_name not in joint_names:
            self.get_logger().warn(
                f"{joint_name} not in trajectory; cannot check wrist sweep."
            )
            return True

        idx = joint_names.index(joint_name)
        positions = [
            point.positions[idx]
            for point in traj_msg.joint_trajectory.points
            if len(point.positions) > idx
        ]
        if not positions:
            self.get_logger().warn(
                f"No positions for {joint_name}; cannot check wrist sweep."
            )
            return True

        delta = max(positions) - min(positions)
        self.get_logger().info(
            f"Cable/wrist check: {joint_name} delta={delta:.3f} rad "
            f"({delta * RAD_TO_DEG:.1f} deg)"
        )

        if abs(delta) > max_delta_rad:
            self.get_logger().error(
                f"Trajectory rejected: {joint_name} rotates too much "
                f"({delta:.3f} rad > {max_delta_rad:.3f} rad)."
            )
            return False

        return True

    def cartesian_plan_to_pose(
        self,
        arm: str,
        pose: PoseStamped,
        pose_link: Optional[str] = None,
        max_step: float = 0.005,
        jump_threshold: float = 0.0,
        min_fraction: float = 0.95,
        timeout_s: float = 10.0,
        check_wrist: bool = True,
    ):
        """Plan a real straight-line Cartesian path to a single pose.

        Unlike arm_go_to_pose_best_of_n (OMPL), this uses /compute_cartesian_path.
        Use it for short descents/lifts (pre-grasp -> grasp, grasp -> lift).
        """

        selected_arm = self._get_arm(arm)
        group_name = selected_arm.planning_group_name
        link = self._get_pose_link(arm, pose_link)

        self.get_logger().info(
            f"Cartesian plan: group='{group_name}', link='{link}', "
            f"frame='{pose.header.frame_id}', max_step={max_step}, "
            f"min_fraction={min_fraction}"
        )

        # Wait for the service. It may show in the graph but not be matched yet
        # (e.g. a stale endpoint after a move_group restart), so we retry and log.
        self.get_logger().info("Waiting for /compute_cartesian_path to be available...")
        service_ready = False
        for attempt in range(1, 16):
            if self.cartesian_plan_client.wait_for_service(timeout_sec=1.0):
                service_ready = True
                break
            visible_services = [
                name
                for name, _types in self.get_service_names_and_types()
                if "compute_cartesian" in name
            ]
            self.get_logger().warn(
                f"/compute_cartesian_path not available yet for this node "
                f"(attempt {attempt}/15). Related visible services: {visible_services}"
            )

        if not service_ready:
            self.get_logger().error(
                "Internal client could not connect to /compute_cartesian_path. "
                "If 'ros2 service list' shows it, it is likely discovery/daemon or a "
                "dead provider endpoint."
            )
            return False, "CARTESIAN_SERVICE_NOT_AVAILABLE", None

        request = GetCartesianPath.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = pose.header.frame_id
        request.group_name = group_name
        request.link_name = link
        request.max_step = max_step
        request.jump_threshold = jump_threshold
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = 0.05
        request.max_acceleration_scaling_factor = 0.05

        with self.planning_monitor.read_write() as scene:
            current_state = scene.current_state
            assert isinstance(current_state, RobotState)
            current_state.update(True)

            # start_state defines where the trajectory begins. We do NOT add the
            # current pose as a waypoint: it may be expressed in another frame and
            # cause fraction=0.000.
            request.start_state = robotStateToRobotStateMsg(current_state)

            # Only the target waypoint; its frame is request.header.frame_id.
            request.waypoints = [pose.pose]

        future = self.cartesian_plan_client.call_async(request)

        start_time = time.monotonic()
        while rclpy.ok() and not future.done():
            if time.monotonic() - start_time > timeout_s:
                self.get_logger().error(
                    f"Timeout waiting for /compute_cartesian_path ({timeout_s}s)."
                )
                return False, "CARTESIAN_TIMEOUT", None
            time.sleep(0.05)

        response = future.result()
        if response is None:
            self.get_logger().error("Empty response from /compute_cartesian_path.")
            return False, "CARTESIAN_EMPTY_RESPONSE", None

        self.get_logger().info(f"Cartesian fraction={response.fraction:.3f}")

        # fraction = portion of the straight path that was solved. Below the
        # threshold means it was blocked (residual voxels, joint limit, singularity).
        if response.fraction < min_fraction:
            self.get_logger().error(
                f"Incomplete Cartesian path: fraction={response.fraction:.3f} "
                f"< min_fraction={min_fraction:.3f}."
            )
            return False, "CARTESIAN_FRACTION_TOO_LOW", None

        robot_trajectory = RobotTrajectory(self.robot_model)
        robot_trajectory.set_robot_trajectory_msg(current_state, response.solution)
        robot_trajectory.joint_model_group_name = group_name

        if check_wrist:
            if not self._trajectory_joint_delta_ok(
                robot_trajectory,
                joint_name="ur_dual_I_wrist_3_joint",
                max_delta_rad=0.35,
            ):
                return False, "WRIST_ROTATION_TOO_LARGE", None

        plan_like_result = SimpleNamespace(trajectory=robot_trajectory)
        self.get_logger().info("CARTESIAN PLAN OK")
        return True, "CARTESIAN_PLAN_SUCCEEDED", plan_like_result
    def arm_go_to_pose_cartesian(
        self,
        arm: str,
        pose: PoseStamped,
        pose_link: Optional[str] = None,
        execute: bool = False,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
    ):
        """Plan (and optionally execute) a real single-pose Cartesian move."""

        success, status, plan_result = self.cartesian_plan_to_pose(
            arm=arm,
            pose=pose,
            pose_link=pose_link,
            max_step=0.005,
            jump_threshold=0.0,
            min_fraction=0.95,
            timeout_s=10.0,
            check_wrist=True,
        )

        if not success or plan_result is None:
            return success, status, None

        if execute:
            exec_result = self.execute_trajectory(
                plan_result.trajectory,
                velocity_scaling=velocity_scaling,
                acceleration_scaling=acceleration_scaling,
            )
            return True, str(exec_result), plan_result

        return True, status, plan_result

    def shutdown(self):
        """Explicit wrapper shutdown."""
        try:
            self.robot.shutdown()
        except Exception:
            pass
