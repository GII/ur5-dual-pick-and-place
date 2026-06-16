#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# ur_dual_command_services.py
#
# Pick-and-place commander (extends UrDualMoveItPy). It exposes the whole cycle
# as ROS 2 services so each step can be planned, inspected in RViz and executed
# separately. It never plans automatically on startup and never blocks on input().
# -----------------------------------------------------------------------------

import time
import math
import rclpy
from rclpy.executors import MultiThreadedExecutor
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, SetBool
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    AllowedCollisionMatrix,
    AllowedCollisionEntry,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

from ur_dual_pick_place.ur_dual_moveit_py import UrDualMoveItPy


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert roll, pitch, yaw to a quaternion (intrinsic XYZ), enough for the
    fixed pre-grasp orientations."""

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy

    return q




# Approximate object sizes (size_x, size_y, size_z) in metres.
OBJECT_DIMENSIONS = {
    "bola":         (0.080, 0.080, 0.080),
    "botella rosa": (0.063, 0.063, 0.200),
    "caballo":      (0.140, 0.140, 0.115),
    "cubo":         (0.062, 0.062, 0.075),
    "lechuga":      (0.070, 0.070, 0.085),
    "pina":         (0.155, 0.155, 0.075),
    "prisma":       (0.125, 0.125, 0.057),
    "refresco":     (0.070, 0.070, 0.220),
    "tomate":       (0.055, 0.055, 0.040),
    "vaca":         (0.125, 0.125, 0.075),
}
OBJECT_DIM_DEFAULT = (0.120, 0.120, 0.140)


# Per-class OrientationConstraint tolerances for the pre-grasp.
# Each tuple is (tol_roll, tol_pitch, tol_yaw) in radians 
GRASP_TOLERANCES = {
    "bola":         (0.10, 0.10, 3.14),   
    "botella rosa": (0.10, 0.10, 3.14),
    "caballo":      (0.10, 0.10, 3.14),
    "cubo":         (0.10, 0.10, 3.14),
    "lechuga":      (1.50, 1.50, 3.14), 
    "pina":         (0.10, 0.10, 3.14),
    "prisma":       (0.10, 0.10, 3.14),
    "refresco":     (0.10, 0.10, 3.14),
    "tomate":       (1.50, 1.50, 3.14),   
    "vaca":         (0.10, 0.10, 3.14),
}
GRASP_TOLERANCE_DEFAULT = (0.10, 0.10, 3.14)



TARGET_COLLISION_ID = "target_object"


class UrDualCommandServices(UrDualMoveItPy):
    def __init__(self, name="ur_dual_commander"):
        super().__init__(name)

        self.pending_plan_result = None
        self.latest_joint_state = None
        self.latest_object_pose = None
        self.last_object_pose_log_time = 0.0

        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10,
        )

        self.create_subscription(
            PoseStamped,
            "/ur_dual/object_pose",
            self._on_object_pose,
            10,
        )

        self.create_subscription(
            String,
            "/ur_dual/object_class",
            self._on_object_class,
            10,
        )

        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10,
        )

        self.hand_pub = self.create_publisher(
            JointTrajectory,
            "/qbhand2m1/qbhand2m1_synergies_trajectory_controller/joint_trajectory",
            10,
        )


        self.planning_scene_pub = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            10,
        )





        # TF: used to read the current tool0 pose and build a relative Cartesian move.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Relative Cartesian move parameters (used by plan_offset_test for
        # descend/lift): move the pose_link by (dx, dy, dz) from its current pose.
        self.declare_parameter("offset_test_base_frame", "ur_dual_I_base_link")
        self.declare_parameter("offset_test_pose_link", "ur_dual_I_tool0")
        self.declare_parameter("offset_test_dx", 0.0)
        self.declare_parameter("offset_test_dy", 0.0)
        self.declare_parameter("offset_test_dz", -0.1)

        # Pre-grasp parameters derived from the detected object pose.
        self.declare_parameter("pregrasp_base_frame", "ur_dual_I_base_link")
        # Pre-grasp for top-grasp objects (palm parallel to the table).
        self.declare_parameter("palm_z_offset", 0.144)
        self.declare_parameter("approach_height", 0.25)
        self.declare_parameter("pregrasp_roll", math.pi)
        self.declare_parameter("pregrasp_pitch", 0.0)
        self.declare_parameter("pregrasp_yaw", 0.0)
        self.declare_parameter("cartesian_safe_z", 1.00)
        self.declare_parameter("cartesian_pregrasp_pose_link", "ur_dual_I_tool0")


        # Grasp mode: "normal" (top grasp) or "bottle" (lateral grasp).
        self.declare_parameter("grasp_mode", "normal")

        # Template for top-grasp objects (object centre in X/Y, Z offset only).
        self.declare_parameter("normal_grasp_dx", 0.0)
        self.declare_parameter("normal_grasp_dy", 0.0)
        self.declare_parameter("normal_grasp_dz", 0.040)

        # Top-grasp reference orientation (not tilted).
        self.declare_parameter("normal_grasp_qx", -0.003)
        self.declare_parameter("normal_grasp_qy", 0.023)
        self.declare_parameter("normal_grasp_qz", -0.022)
        self.declare_parameter("normal_grasp_qw", 0.999)

        # Template for bottles (lateral approach; tune the offsets per setup).
        self.declare_parameter("bottle_grasp_dx", 0.0)
        self.declare_parameter("bottle_grasp_dy", 0.0)
        self.declare_parameter("bottle_grasp_dz", 0.0)

        # Averaged bottle grasp orientation.
        self.declare_parameter("bottle_grasp_qx", -0.008)
        self.declare_parameter("bottle_grasp_qy", 0.680)
        self.declare_parameter("bottle_grasp_qz", 0.028)
        self.declare_parameter("bottle_grasp_qw", 0.733)

        self.declare_parameter("pregrasp_pose_link", "qbhand2m1_palm_link")
        self.declare_parameter("pregrasp_orientation_mode", "grasp_template")

        # Attached object geometry. The grasped object is modelled as a
        # conservative box attached to the hand; one fixed size per category
        # (a dynamic bbox from the point cloud would be future work).
        self.declare_parameter("attached_object_id", "grasped_object")
        self.declare_parameter("attached_object_frame", "qbhand2m1_palm_link")
        # Box size in metres (width, depth, height).
        self.declare_parameter("attached_object_size_x", 0.08)
        self.declare_parameter("attached_object_size_y", 0.08)
        self.declare_parameter("attached_object_size_z", 0.12)
        # Offset of the box centre relative to the hand frame (a grasped object
        # usually sits slightly in front of palm_link).
        self.declare_parameter("attached_object_offset_x", 0.0)
        self.declare_parameter("attached_object_offset_y", 0.0)
        self.declare_parameter("attached_object_offset_z", -0.05)
        # Extra per-axis margin when clearing octomap voxels around the object;
        # absorbs pose-vs-surface offset up to this value (m).
        self.declare_parameter("clear_box_margin", 0.02)
        


        self.create_service(
            Trigger,
            "/ur_dual/plan_home_right",
            self._on_plan_home_right,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_offset_test",
            self._on_plan_offset_test,
        )


        self.create_service(
            Trigger,
            "/ur_dual/plan_pregrasp_from_latest_pose",
            self._on_plan_pregrasp_from_latest_pose,
        )

        self.create_service(
            SetBool,
            "/ur_dual/execute_last_plan",
            self._on_execute_last_plan,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_ready_right",
            self._on_plan_ready_right,
        )

        self.create_service(
            Trigger,
            "/ur_dual/open_hand",
            self._on_open_hand,
        )

        self.create_service(
            Trigger,
            "/ur_dual/close_hand",
            self._on_close_hand,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_place_normal",
            self._on_plan_place_normal,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_place_bottle",
            self._on_plan_place_bottle,
        )

        self.create_service(
            Trigger,
            "/ur_dual/clear_octomap",
            self._on_clear_octomap,
        )



        # FREEZE / UNFREEZE the OctoMap
        self.create_service(
            Trigger,
            "/ur_dual/freeze_octomap",
            self._on_freeze_octomap,
        )
        self.create_service(
            Trigger,
            "/ur_dual/unfreeze_octomap",
            self._on_unfreeze_octomap,
        )

        # Clients for the OAK stop/start services.
        self.oak_stop_client = self.create_client(
            Trigger,
            "/oak_cam/oak/stop_camera",
            callback_group=self.cb_internal_client,
        )
        self.oak_start_client = self.create_client(
            Trigger,
            "/oak_cam/oak/start_camera",
            callback_group=self.cb_internal_client,
        )



        # Attach / detach the grasped object. attach: right after a successful
        # close_hand; detach: right after open_hand 
        self.create_service(
            Trigger,
            "/ur_dual/clear_octomap_around_object",
            self._on_clear_octomap_around_object,
        )
        self.create_service(
            Trigger,
            "/ur_dual/attach_grasped_object",
            self._on_attach_grasped_object,
        )
        self.create_service(
            Trigger,
            "/ur_dual/detach_grasped_object",
            self._on_detach_grasped_object,
        )



        # True while the OctoMap is frozen (camera stopped).
        self._octomap_frozen = False
        # True while an object is attached to the hand.
        self._object_attached = False

        # Latest detected object class (from object_pose_bridge). Drives the
        # clear box size and the pre-grasp orientation tolerances.
        self._latest_object_class = None



        self.get_logger().info(
            "Pick-and-place services ready. Plan with a service, inspect in\n"
            "RViz, then execute with /ur_dual/execute_last_plan.\n"
            "  pregrasp : /ur_dual/plan_pregrasp_from_latest_pose\n"
            "  descend/lift : /ur_dual/plan_offset_test (set offset_test_dz)\n"
            "  hand     : /ur_dual/close_hand , /ur_dual/open_hand\n"
            "  attach   : /ur_dual/attach_grasped_object , detach_grasped_object\n"
            "  named    : /ur_dual/plan_ready_right , plan_place_normal , plan_place_bottle\n"
            "  octomap  : /ur_dual/freeze_octomap , unfreeze_octomap , clear_octomap_around_object\n"
            "  execute  : /ur_dual/execute_last_plan {data: true|false}"
        )

    def _on_joint_state(self, msg: JointState):
        self.latest_joint_state = msg


    def _on_object_pose(self, msg: PoseStamped):
        """Cache the latest object pose (already in the robot base frame). Logging
        is rate-limited because the bridge publishes detections continuously."""

        if not msg.header.frame_id:
            self.get_logger().warn("Pose de objeto recibida sin frame_id. Se ignora.")
            return

        self.latest_object_pose = msg

        now = time.monotonic()
        if now - self.last_object_pose_log_time < 1.5:
            return

        self.last_object_pose_log_time = now
        self.get_logger().info(
            "Pose de objeto recibida:\n"
            f"  frame_id: {msg.header.frame_id}\n"
            f"  x={msg.pose.position.x:.3f}, "
            f"y={msg.pose.position.y:.3f}, "
            f"z={msg.pose.position.z:.3f}"
        )


    def _wait_for_joint_state(self, timeout_s=5.0) -> bool:
        start = time.monotonic()

        while time.monotonic() - start < timeout_s:
            if self.latest_joint_state is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    def _publish_plan_for_rviz(self, plan_result, n_times=5, period_s=0.5):
        if self.latest_joint_state is None:
            self.get_logger().warn(
                "No hay /joint_states todavía. Publicaré trayectoria sin trajectory_start."
            )
            trajectory_start = RobotState()
        else:
            trajectory_start = RobotState()
            trajectory_start.joint_state = self.latest_joint_state
            trajectory_start.is_diff = False

        traj_msg = plan_result.trajectory.get_robot_trajectory_msg()

        display_msg = DisplayTrajectory()
        display_msg.model_id = "ur_dual"
        display_msg.trajectory_start = trajectory_start
        display_msg.trajectory.append(traj_msg)

        for i in range(n_times):
            self.display_pub.publish(display_msg)
            self.get_logger().info(
                f"Plan publicado en /display_planned_path ({i + 1}/{n_times})."
            )
            time.sleep(period_s)


    def _make_pregrasp_pose_from_latest_object(self) -> PoseStamped | None:
        """Build a pre-grasp pose from the latest object pose: position =
        object pose + per-mode offset, orientation = the taught template."""

        if self.latest_object_pose is None:
            self.get_logger().error(
                "No hay pose de objeto todavía. Espera a que /ur_dual/object_pose publique."
            )
            return None

        object_pose = self.latest_object_pose

        base_frame = self.get_parameter("pregrasp_base_frame").value
        pose_link = self.get_parameter("pregrasp_pose_link").value
        orientation_mode = self.get_parameter("pregrasp_orientation_mode").value

        if object_pose.header.frame_id != base_frame:
            self.get_logger().error(
                f"La pose de objeto debe venir en '{base_frame}', "
                f"pero llegó en '{object_pose.header.frame_id}'."
            )
            return None

        pregrasp = PoseStamped()
        pregrasp.header.frame_id = base_frame
        pregrasp.header.stamp = self.get_clock().now().to_msg()


        if orientation_mode in ("normal_grasp", "grasp_template"):
            grasp_mode = self.get_parameter("grasp_mode").value

            if grasp_mode == "bottle":
                prefix = "bottle_grasp"
                position_mode = "BOTTLE_GRASP_TEMPLATE"
            else:
                prefix = "normal_grasp"
                position_mode = "NORMAL_GRASP_TEMPLATE"

            dx = float(self.get_parameter(f"{prefix}_dx").value)
            dy = float(self.get_parameter(f"{prefix}_dy").value)
            dz = float(self.get_parameter(f"{prefix}_dz").value)

            pregrasp.pose.position.x = object_pose.pose.position.x + dx
            pregrasp.pose.position.y = object_pose.pose.position.y + dy
            pregrasp.pose.position.z = object_pose.pose.position.z + dz

            pregrasp.pose.orientation.x = float(
                self.get_parameter(f"{prefix}_qx").value
            )
            pregrasp.pose.orientation.y = float(
                self.get_parameter(f"{prefix}_qy").value
            )
            pregrasp.pose.orientation.z = float(
                self.get_parameter(f"{prefix}_qz").value
            )
            pregrasp.pose.orientation.w = float(
                self.get_parameter(f"{prefix}_qw").value
            )

            position_mode = (
                f"{position_mode}: object + "
                f"dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}"
            )

        # Fallback: use the link's current orientation.
        elif orientation_mode == "current_tf":
            approach_height = float(self.get_parameter("approach_height").value)

            pregrasp.pose.position.x = object_pose.pose.position.x
            pregrasp.pose.position.y = object_pose.pose.position.y
            pregrasp.pose.position.z = object_pose.pose.position.z + approach_height

            try:
                transform = self.tf_buffer.lookup_transform(
                    base_frame,
                    pose_link,
                    Time(),
                    timeout=Duration(seconds=2.0),
                )
                pregrasp.pose.orientation = transform.transform.rotation
            except Exception as exc:
                self.get_logger().error(
                    f"No se pudo leer TF {base_frame} <- {pose_link}: {exc}"
                )
                return None

            position_mode = "CURRENT_TF: object + approach_height"

        # Fallback: orientation from RPY.
        else:
            palm_z_offset = float(self.get_parameter("palm_z_offset").value)
            approach_height = float(self.get_parameter("approach_height").value)

            pregrasp.pose.position.x = object_pose.pose.position.x
            pregrasp.pose.position.y = object_pose.pose.position.y

            if pose_link == "qbhand2m1_palm_link":
                pregrasp.pose.position.z = object_pose.pose.position.z + approach_height
                position_mode = "PALM_LINK_RPY: object.z + approach_height"
            else:
                pregrasp.pose.position.z = (
                    object_pose.pose.position.z + palm_z_offset + approach_height
                )
                position_mode = "TOOL0_RPY: object.z + palm_z_offset + approach_height"

            roll = float(self.get_parameter("pregrasp_roll").value)
            pitch = float(self.get_parameter("pregrasp_pitch").value)
            yaw = float(self.get_parameter("pregrasp_yaw").value)

            pregrasp.pose.orientation = quaternion_from_rpy(roll, pitch, yaw)

        q = pregrasp.pose.orientation

        self.get_logger().info(
            "Pose pre-grasp generada:\n"
            f"  frame_id: {pregrasp.header.frame_id}\n"
            f"  pose_link: {pose_link}\n"
            f"  x={pregrasp.pose.position.x:.3f}, "
            f"y={pregrasp.pose.position.y:.3f}, "
            f"z={pregrasp.pose.position.z:.3f}\n"
            f"  objeto x={object_pose.pose.position.x:.3f}, "
            f"y={object_pose.pose.position.y:.3f}, "
            f"z={object_pose.pose.position.z:.3f}\n"
            f"  position_mode={position_mode}\n"
            f"  orientation_mode={orientation_mode}\n"
            f"  quat=[x={q.x:.3f}, y={q.y:.3f}, z={q.z:.3f}, w={q.w:.3f}]"
        )

        return pregrasp


    def _make_offset_pose_from_current_tool(self):
        """Read the current pose_link pose via TF and return a target pose shifted
        by (offset_test_d{x,y,z}). Orientation is kept; used by plan_offset_test."""

        base_frame = self.get_parameter("offset_test_base_frame").value
        pose_link = self.get_parameter("offset_test_pose_link").value

        dx = float(self.get_parameter("offset_test_dx").value)
        dy = float(self.get_parameter("offset_test_dy").value)
        dz = float(self.get_parameter("offset_test_dz").value)

        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                pose_link,
                Time(),
                timeout=Duration(seconds=2.0),
            )
        except Exception as exc:
            self.get_logger().error(
                f"No se pudo leer TF {base_frame} <- {pose_link}: {exc}"
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = base_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = transform.transform.translation.x + dx
        pose.pose.position.y = transform.transform.translation.y + dy
        pose.pose.position.z = transform.transform.translation.z + dz

        # Keep tool0's current orientation to avoid odd IK or unexpected twists.
        pose.pose.orientation = transform.transform.rotation

        self.get_logger().info(
            "Pose offset generada:\n"
            f"  frame_id: {pose.header.frame_id}\n"
            f"  pose_link: {pose_link}\n"
            f"  posición objetivo: "
            f"x={pose.pose.position.x:.3f}, "
            f"y={pose.pose.position.y:.3f}, "
            f"z={pose.pose.position.z:.3f}\n"
            f"  offset aplicado: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}"
        )

        return pose


    def _on_plan_pregrasp_from_latest_pose(self, request, response):
        """Plan (OMPL, best-of-N, cable-protected) from the current state to the
        pre-grasp pose. Does not execute; leaves a pending plan for RViz review."""

        self.get_logger().info("Servicio recibido: plan_pregrasp_from_latest_pose")
        
        # Drop any previous plan so a stale trajectory cannot be executed.
        self.pending_plan_result = None

        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = (
                "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            )
            return response

        pregrasp_pose = self._make_pregrasp_pose_from_latest_object()

        if pregrasp_pose is None:
            response.success = False
            response.message = "No se pudo generar la pose pre-grasp."
            return response

        pose_link = self.get_parameter("pregrasp_pose_link").value




        # OrientationConstraint tolerances by YOLO class (default: strict
        # roll/pitch, free yaw).
        obj_class = self._latest_object_class
        if obj_class in GRASP_TOLERANCES:
            tol_roll, tol_pitch, tol_yaw = GRASP_TOLERANCES[obj_class]
            tol_src = f"clase '{obj_class}'"
        else:
            tol_roll, tol_pitch, tol_yaw = GRASP_TOLERANCE_DEFAULT
            tol_src = f"default (clase '{obj_class}' no en GRASP_TOLERANCES)"

        self.get_logger().info(
            f"Pregrasp con OrientationConstraint [{tol_src}]: "
            f"tol=({tol_roll:.2f},{tol_pitch:.2f},{tol_yaw:.2f}) rad"
        )



        success, status, plan_result = self.arm_go_to_pose_best_of_n(
            arm="right",
            pose=pregrasp_pose,
            pose_link=pose_link,
            attempts=12,
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
            orientation_tolerance=(tol_roll, tol_pitch, tol_yaw),
            position_tolerance=0.005,
        )

        if not success:
            self.pending_plan_result = None

            # MoveIt found a trajectory but the cable filter rejected it. Publish
            # it to RViz for diagnosis only; it is NOT left pending for execution.
            if status == "CABLE_MOTION_TOO_LARGE" and plan_result is not None:
                self.get_logger().warn(
                    "El plan fue rechazado por protección de cable, "
                    "pero se publicará en RViz solo para diagnóstico. "
                    "NO queda pendiente para ejecución."
                )

                self._publish_plan_for_rviz(plan_result, n_times=5, period_s=0.5)

                response.success = False
                response.message = (
                    "Plan encontrado pero rechazado por cable. "
                    "Publicado en RViz solo para diagnóstico. NO ejecutar."
                )
                return response

            response.success = False
            response.message = f"Plan pre-grasp falló. Status: {status}"
            return response

        if plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = "Plan pre-grasp falló: plan_result vacío."
            return response

        self.pending_plan_result = plan_result

        # Few repeats so the service responds quickly.
        self._publish_plan_for_rviz(plan_result, n_times=3, period_s=0.2)

        response.success = True
        response.message = (
            "Plan pre-grasp generado y publicado en RViz. "
            "Revisa que los joints no giren demasiado antes de ejecutar."
        )
        return response


    def _on_plan_offset_test(self, request, response):
        """Straight-line Cartesian move to current_pose + (offset_test_d{x,y,z}).
        Used for descend/lift. Plans only; does not execute."""

        self.get_logger().info("Servicio recibido: plan_offset_test")
        # Drop any previous plan when starting a new one.
        self.pending_plan_result = None

        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = (
                "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            )
            return response

        pose = self._make_offset_pose_from_current_tool()

        if pose is None:
            response.success = False
            response.message = "No se pudo generar la pose offset por TF."
            return response

        pose_link = self.get_parameter("offset_test_pose_link").value

        success, status, plan_result = self.arm_go_to_pose_cartesian(
            arm="right",
            pose=pose,
            pose_link=pose_link,
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        if not success or plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = f"Plan cartesiano offset falló. Status: {status}"
            return response

        self.pending_plan_result = plan_result
        self._publish_plan_for_rviz(plan_result, n_times=5, period_s=0.5)

        response.success = True
        response.message = (
            "Plan offset generado y publicado en RViz. "
            "Revisa la trayectoria antes de ejecutar."
        )
        return response


    def _on_plan_home_right(self, request, response):
        self.get_logger().info("Servicio recibido: plan_home_right")
        # Drop any previous plan when starting a new one.
        self.pending_plan_result = None
        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            return response

        success, status, plan_result = self.arm_go_to_named_pose(
            arm="right",
            pose_name="Home_Right",
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        if not success or plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = f"Plan falló. Status: {status}"
            return response

        self.pending_plan_result = plan_result
        self._publish_plan_for_rviz(plan_result, n_times=5, period_s=0.5)

        response.success = True
        response.message = (
            "Plan hacia Home_Right generado y publicado en RViz. "
            "Revisa la trayectoria antes de ejecutar."
        )
        return response

    def _on_plan_ready_right(self, request, response):
        """Plan to the SRDF Ready_Right named pose."""

        self.get_logger().info("Servicio recibido: plan_ready_right")

        self.pending_plan_result = None

        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = (
                "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            )
            return response

        success, status, plan_result = self.arm_go_to_named_pose(
            arm="right",
            pose_name="Ready_Right",
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        if not success or plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = f"Plan Ready_Right falló. Status: {status}"
            return response

        self.pending_plan_result = plan_result
        self._publish_plan_for_rviz(plan_result, n_times=3, period_s=0.2)

        response.success = True
        response.message = (
            "Plan Ready_Right generado y publicado en RViz. "
            "Revisa antes de ejecutar."
        )
        return response



    def _publish_hand_command(
        self,
        synergy: float,
        manipulation: float,
        duration_s: int = 2,
    ):
        msg = JointTrajectory()
        msg.joint_names = [
            "qbhand2m1_synergy_joint",
            "qbhand2m1_manipulation_joint",
        ]

        point = JointTrajectoryPoint()
        point.positions = [synergy, manipulation]
        point.time_from_start.sec = duration_s

        msg.points.append(point)
        self.hand_pub.publish(msg)

    def _on_open_hand(self, request, response):
        self.get_logger().info("Servicio recibido: open_hand")
        self._publish_hand_command(synergy=0.0, manipulation=0.0, duration_s=2)
        response.success = True
        response.message = "Comando de abrir mano publicado."
        return response

    def _on_close_hand(self, request, response):
        self.get_logger().info("Servicio recibido: close_hand")
        self._publish_hand_command(synergy=0.85, manipulation=0.0, duration_s=2)
        response.success = True
        response.message = "Comando de cerrar mano publicado."
        return response

    def _plan_named_pose_service(self, pose_name: str, response):
        self.pending_plan_result = None

        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = (
                "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            )
            return response

        success, status, plan_result = self.arm_go_to_named_pose(
            arm="right",
            pose_name=pose_name,
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        if not success or plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = f"Plan {pose_name} falló. Status: {status}"
            return response

        self.pending_plan_result = plan_result
        self._publish_plan_for_rviz(plan_result, n_times=3, period_s=0.2)

        response.success = True
        response.message = f"Plan {pose_name} generado y publicado en RViz."
        return response

    def _on_plan_place_normal(self, request, response):
        self.get_logger().info("Servicio recibido: plan_place_normal")
        return self._plan_named_pose_service("Place_Normal_Right", response)

    def _on_plan_place_bottle(self, request, response):
        self.get_logger().info("Servicio recibido: plan_place_bottle")
        return self._plan_named_pose_service("Place_Bottle_Right", response)





    def _call_oak_service(self, client, service_label, timeout_s=3.0):
        """Call an OAK Trigger service with a timeout."""
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                f"Servicio {service_label} no disponible (cámara no corriendo?)."
            )
            return False

        req = Trigger.Request()
        future = client.call_async(req)

        start = time.monotonic()
        while rclpy.ok() and not future.done():
            if time.monotonic() - start > timeout_s:
                self.get_logger().error(f"Timeout llamando a {service_label}.")
                return False
            time.sleep(0.05)

        result = future.result()
        if result is None or not result.success:
            self.get_logger().error(
                f"{service_label} respondió fallo: "
                f"{result.message if result else 'sin respuesta'}"
            )
            return False

        return True

    def _on_freeze_octomap(self, request, response):
        """Freeze the OctoMap by stopping the OAK pipeline, so the moving arm,
        hand and cable are not added as spurious obstacles during execution."""
        if self._octomap_frozen:
            response.success = True
            response.message = "OctoMap ya estaba congelado."
            return response

        self.get_logger().info(
            "FREEZE OctoMap: deteniendo pipeline de la OAK..."
        )

        ok = self._call_oak_service(self.oak_stop_client, "/oak_cam/oak/stop_camera")
        if not ok:
            response.success = False
            response.message = "No pude detener la cámara para congelar OctoMap."
            return response

        # Small delay so the updater processes the last queued frames.
        time.sleep(0.5)

        self._octomap_frozen = True
        response.success = True
        response.message = "OctoMap congelado (cámara detenida)."
        self.get_logger().info(response.message)
        return response

    def _on_unfreeze_octomap(self, request, response):
        """Resume OctoMap capture by restarting the OAK pipeline. Called at the
        end of the cycle, once the arm is back at Ready_Right."""
        if not self._octomap_frozen:
            response.success = True
            response.message = "OctoMap ya estaba activo."
            return response

        self.get_logger().info(
            "UNFREEZE OctoMap: rearrancando pipeline de la OAK..."
        )

        ok = self._call_oak_service(self.oak_start_client, "/oak_cam/oak/start_camera")
        if not ok:
            response.success = False
            response.message = "No pude rearrancar la cámara para descongelar OctoMap."
            return response

        # The OAK takes ~1-2 s to stabilise after start; we do not block here.
        self._octomap_frozen = False
        response.success = True
        response.message = "OctoMap descongelado (cámara reiniciada)."
        self.get_logger().info(response.message)
        return response






    def _get_softhand_touch_links(self):
        """SoftHand links the attached object is allowed to touch without being
        flagged as a collision (touch_links). Names come from the qbhand2m1 URDF."""
        return [
            "qbhand2m1_palm_link",
            "qbhand2m1_thumb_knuckle_link",
            "qbhand2m1_thumb_proximal_link",
            "qbhand2m1_thumb_distal_link",
            "qbhand2m1_index_knuckle_link",
            "qbhand2m1_index_proximal_link",
            "qbhand2m1_index_middle_link",
            "qbhand2m1_index_distal_link",
            "qbhand2m1_middle_knuckle_link",
            "qbhand2m1_middle_proximal_link",
            "qbhand2m1_middle_middle_link",
            "qbhand2m1_middle_distal_link",
            "qbhand2m1_ring_knuckle_link",
            "qbhand2m1_ring_proximal_link",
            "qbhand2m1_ring_middle_link",
            "qbhand2m1_ring_distal_link",
            "qbhand2m1_little_knuckle_link",
            "qbhand2m1_little_proximal_link",
            "qbhand2m1_little_middle_link",
            "qbhand2m1_little_distal_link",
        ]

    def _on_attach_grasped_object(self, request, response):
        """Attach the grasped object as an AttachedCollisionObject on the hand
        frame, so MoveIt checks hand+object against obstacles during lift/place."""
        if self._object_attached:
            response.success = True
            response.message = "El objeto ya estaba attached."
            return response

        obj_id = self.get_parameter("attached_object_id").value
        frame = self.get_parameter("attached_object_frame").value
        sx = float(self.get_parameter("attached_object_size_x").value)
        sy = float(self.get_parameter("attached_object_size_y").value)
        sz = float(self.get_parameter("attached_object_size_z").value)
        ox = float(self.get_parameter("attached_object_offset_x").value)
        oy = float(self.get_parameter("attached_object_offset_y").value)
        oz = float(self.get_parameter("attached_object_offset_z").value)

        self.get_logger().info(
            f"ATTACH '{obj_id}' a '{frame}': "
            f"size=({sx:.3f},{sy:.3f},{sz:.3f}), offset=({ox:.3f},{oy:.3f},{oz:.3f})"
        )

        # Base CollisionObject (a box).
        collision = CollisionObject()
        collision.header.frame_id = frame
        collision.header.stamp = self.get_clock().now().to_msg()
        collision.id = obj_id
        collision.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [sx, sy, sz]

        pose = Pose()
        pose.position.x = ox
        pose.position.y = oy
        pose.position.z = oz
        pose.orientation.w = 1.0

        collision.primitives.append(box)
        collision.primitive_poses.append(pose)

        # Wrap it as an AttachedCollisionObject.
        attached = AttachedCollisionObject()
        attached.link_name = frame
        attached.object = collision
        attached.touch_links = self._get_softhand_touch_links()

        # Publish to the planning scene as a diff.
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)

        self.planning_scene_pub.publish(scene)

        # Small delay so the PlanningSceneMonitor processes the diff.
        time.sleep(0.3)

        self._object_attached = True
        response.success = True
        response.message = f"Objeto '{obj_id}' attached a '{frame}'."
        self.get_logger().info(response.message)
        return response

    def _on_detach_grasped_object(self, request, response):
        """Detach the object from the hand. Call right before opening the hand at
        the place; afterwards MoveIt no longer treats it as part of the robot."""
        if not self._object_attached:
            response.success = True
            response.message = "No hay objeto attached para liberar."
            return response

        obj_id = self.get_parameter("attached_object_id").value
        frame = self.get_parameter("attached_object_frame").value

        self.get_logger().info(f"DETACH '{obj_id}' de '{frame}'.")

        # Detach via AttachedCollisionObject REMOVE.
        collision = CollisionObject()
        collision.id = obj_id
        collision.operation = CollisionObject.REMOVE
        collision.header.frame_id = frame

        attached = AttachedCollisionObject()
        attached.link_name = frame
        attached.object = collision

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)

        self.planning_scene_pub.publish(scene)

        # Also remove it from the world in case it remains floating.
        time.sleep(0.2)
        world_remove = CollisionObject()
        world_remove.id = obj_id
        world_remove.operation = CollisionObject.REMOVE
        world_remove.header.frame_id = frame

        scene2 = PlanningScene()
        scene2.is_diff = True
        scene2.world.collision_objects.append(world_remove)
        self.planning_scene_pub.publish(scene2)

        time.sleep(0.3)

        self._object_attached = False
        response.success = True
        response.message = f"Objeto '{obj_id}' detached."
        self.get_logger().info(response.message)
        return response




    def _on_object_class(self, msg):
        self._latest_object_class = msg.data

    def _on_clear_octomap_around_object(self, request, response):
        """Clear OctoMap voxels inside a box sized to the detected object's class
        (OBJECT_DIMENSIONS) and centred on its pose. MoveIt removes the voxels that
        fall inside; the 'object does not exist' warning it logs is harmless. Call
        AFTER freeze_octomap so the cleared voxels are not repopulated."""
        if self.latest_object_pose is None:
            response.success = False
            response.message = "No hay pose de objeto disponible."
            return response

        margin = float(self.get_parameter("clear_box_margin").value)

        obj_class = self._latest_object_class
        if obj_class in OBJECT_DIMENSIONS:
            dx, dy, dz = OBJECT_DIMENSIONS[obj_class]
            src = f"clase '{obj_class}'"
        else:
            dx, dy, dz = OBJECT_DIM_DEFAULT
            src = f"default (clase '{obj_class}' no en diccionario)"

        sx = dx + 2.0 * margin
        sy = dy + 2.0 * margin
        sz = dz + 2.0 * margin

        pose = self.latest_object_pose

        self.get_logger().info(
            f"CLEAR OctoMap caja para {src}: "
            f"size=({sx:.3f},{sy:.3f},{sz:.3f}) m centrada en "
            f"({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},"
            f"{pose.pose.position.z:.3f})"
        )

        # BOX CollisionObject with REMOVE op: MoveIt clears the voxels inside it.
        collision = CollisionObject()
        collision.header.frame_id = pose.header.frame_id
        collision.header.stamp = self.get_clock().now().to_msg()
        collision.id = "octomap_clear_region"
        collision.operation = CollisionObject.REMOVE

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [sx, sy, sz]

        box_pose = Pose()
        box_pose.position = pose.pose.position
        box_pose.orientation.w = 1.0

        collision.primitives.append(box)
        collision.primitive_poses.append(box_pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(collision)

        self.planning_scene_pub.publish(scene)
        time.sleep(0.3)

        response.success = True
        response.message = (
            f"Limpieza solicitada ({src}): caja "
            f"{sx:.3f}x{sy:.3f}x{sz:.3f}m."
        )
        return response








    def _on_clear_octomap(self, request, response):
        """Clear this commander's local OctoMap."""

        self.get_logger().info("Servicio recibido: clear_octomap local del commander")

        success, message = self.clear_local_octomap()

        response.success = success
        response.message = message
        return response





    def _on_execute_last_plan(self, request, response):
        if request.data is False:
            self.pending_plan_result = None
            response.success = True
            response.message = "Plan pendiente descartado."
            return response

        if self.pending_plan_result is None:
            response.success = False
            response.message = "No hay plan pendiente. Primero llama un servicio de planificación."
            return response

        plan_to_execute = self.pending_plan_result

        # Clear the pending plan immediately so a stale trajectory is not reused.
        self.pending_plan_result = None

        self.get_logger().warn(
            "Ejecutando último plan. Vigila el área de trabajo y el paro de emergencia."
        )

        result = self.execute_trajectory(
            plan_to_execute.trajectory,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        status_text = str(getattr(result, "status", result))

        response.success = "SUCCEEDED" in status_text or "SUCCESS" in status_text or bool(result)
        response.message = f"Resultado ejecución: {status_text}"

        return response


def main():
    rclpy.init()

    node = UrDualCommandServices(name="ur_dual_commander")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
