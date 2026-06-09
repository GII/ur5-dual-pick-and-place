#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# TFG Daniel Rodríguez Rivas — Pre-grasp dinámico desde visión usando MoveItPy.
#
# Entrada:
#   /object_tracker/detections_base  (ObjDetArray en ur_dual_I_base_link)
#
# Salida:
#   Planificación MoveIt para llevar un link objetivo, normalmente
#   qbhand2m1_palm_link o ur_dual_I_tool0, a una pose sobre el objeto detectado.
#
# Decisión:
#   Se usa MoveItPy en lugar de ActionClient manual sobre /move_action.
# -----------------------------------------------------------------------------

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from stereo_location_interfaces.msg import ObjDetArray
from moveit_msgs.msg import DisplayTrajectory
from moveit.planning import MoveItPy


def quaternion_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class VisualPregraspMoveItPy(Node):
    def __init__(self):
        super().__init__("visual_pregrasp_moveitpy")

        self.declare_parameter("planning_group", "Right_arm")
        self.declare_parameter("pose_link", "qbhand2m1_palm_link")
        self.declare_parameter("base_frame", "ur_dual_I_base_link")
        self.declare_parameter("detections_topic", "/object_tracker/detections_base")
        self.declare_parameter("target_class", "vaca")
        self.declare_parameter("min_confidence", 0.6)

        # Si pose_link = qbhand2m1_palm_link, palm_z_offset debe ser 0.0.
        # Si pose_link = ur_dual_I_tool0, se puede usar 0.144 aprox.
        self.declare_parameter("palm_z_offset", 0.0)
        self.declare_parameter("approach_height", 0.30)

        # Orientación del link objetivo.
        self.declare_parameter("orientation_roll", 0.0)
        self.declare_parameter("orientation_pitch", 0.0)
        self.declare_parameter("orientation_yaw", 0.0)

        self.declare_parameter("plan_only", True)

        self.planning_group = self.get_parameter("planning_group").value
        self.pose_link = self.get_parameter("pose_link").value
        self.base_frame = self.get_parameter("base_frame").value
        self.detections_topic = self.get_parameter("detections_topic").value

        self.executing = False
        self.done = False
        self.latest_joint_state = None

        self.get_logger().info("Inicializando MoveItPy...")
        self.moveit = MoveItPy(node_name="moveit_py_internal")
        self.arm = self.moveit.get_planning_component(self.planning_group)

        self.get_logger().info(
            f"MoveItPy listo | group={self.planning_group} | pose_link={self.pose_link}"
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joint_states,
            10,
        )

        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10,
        )

        self.sub = self.create_subscription(
            ObjDetArray,
            self.detections_topic,
            self.on_detections,
            10,
        )

        self.get_logger().info(
            f"Esperando detecciones en {self.detections_topic} | "
            f"target_class={self.get_parameter('target_class').value} | "
            f"plan_only={self.get_parameter('plan_only').value}"
        )

    def on_joint_states(self, msg: JointState):
        self.latest_joint_state = msg

    def on_detections(self, msg: ObjDetArray):
        if self.executing or self.done:
            return

        target_class = self.get_parameter("target_class").value.strip().lower()
        min_conf = float(self.get_parameter("min_confidence").value)

        candidates = []
        for det in msg.objects:
            if det.class_name.strip().lower() != target_class:
                continue
            if det.confidence < min_conf:
                continue
            if not (
                math.isfinite(det.position.x)
                and math.isfinite(det.position.y)
                and math.isfinite(det.position.z)
            ):
                continue
            candidates.append(det)

        if not candidates:
            return

        target = max(candidates, key=lambda d: d.confidence)

        self.executing = True
        try:
            self.execute_pregrasp(target)
        finally:
            self.done = True
            self.executing = False
            self.get_logger().info(
                "Ciclo terminado. Nodo queda vivo para revisar RViz. "
                "Presiona Ctrl+C para cerrar."
            )

    def execute_pregrasp(self, target):
        palm_z_offset = float(self.get_parameter("palm_z_offset").value)
        approach_height = float(self.get_parameter("approach_height").value)
        plan_only = bool(self.get_parameter("plan_only").value)

        roll = float(self.get_parameter("orientation_roll").value)
        pitch = float(self.get_parameter("orientation_pitch").value)
        yaw = float(self.get_parameter("orientation_yaw").value)

        pose_goal = PoseStamped()
        pose_goal.header.frame_id = self.base_frame
        pose_goal.header.stamp = self.get_clock().now().to_msg()

        pose_goal.pose.position.x = float(target.position.x)
        pose_goal.pose.position.y = float(target.position.y)
        pose_goal.pose.position.z = float(
            target.position.z + palm_z_offset + approach_height
        )
        pose_goal.pose.orientation = quaternion_from_rpy(roll, pitch, yaw)

        self.get_logger().info(
            f"Objetivo seleccionado: {target.class_name} | conf={target.confidence:.3f}"
        )
        self.get_logger().info(
            f"Pose goal para {self.pose_link}: "
            f"x={pose_goal.pose.position.x:.3f}, "
            f"y={pose_goal.pose.position.y:.3f}, "
            f"z={pose_goal.pose.position.z:.3f}, "
            f"rpy=({math.degrees(roll):.1f}, "
            f"{math.degrees(pitch):.1f}, "
            f"{math.degrees(yaw):.1f})"
        )

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(
            pose_stamped_msg=pose_goal,
            pose_link=self.pose_link,
        )

        self.get_logger().info("Planificando con MoveItPy...")
        plan_result = self.arm.plan()

        if not plan_result:
            self.get_logger().error("MoveItPy: planificación falló.")
            return

        self.get_logger().info("✓ MoveItPy: plan encontrado.")
        self.publish_display_trajectory(plan_result)

        if plan_only:
            self.get_logger().info("plan_only=True: NO ejecuto el robot físico.")
            return

        self.get_logger().warn("plan_only=False: ejecutando trayectoria en robot.")
        robot_trajectory = plan_result.trajectory
        self.moveit.execute(robot_trajectory, controllers=[])
        self.get_logger().info("✓ Ejecución terminada.")

    def publish_display_trajectory(self, plan_result):
        """Publica el plan de MoveItPy en /display_planned_path para verlo en RViz."""
        trajectory_obj = getattr(plan_result, "trajectory", None)

        if trajectory_obj is None:
            self.get_logger().warn(
                "No pude publicar en RViz: plan_result no tiene atributo 'trajectory'."
            )
            return

        # MoveItPy devuelve un objeto interno RobotTrajectory.
        # DisplayTrajectory necesita el mensaje ROS moveit_msgs/msg/RobotTrajectory.
        if hasattr(trajectory_obj, "get_robot_trajectory_msg"):
            trajectory_msg = trajectory_obj.get_robot_trajectory_msg()
        else:
            trajectory_msg = trajectory_obj

        display_msg = DisplayTrajectory()
        display_msg.model_id = "ur_dual"

        if self.latest_joint_state is not None:
            display_msg.trajectory_start.joint_state = self.latest_joint_state
            display_msg.trajectory_start.is_diff = True

        display_msg.trajectory.append(trajectory_msg)
        self.display_pub.publish(display_msg)

        self.get_logger().info("Plan publicado en /display_planned_path para RViz.")


def main(args=None):
    rclpy.init(args=args)
    node = VisualPregraspMoveItPy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
