#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# TFG Daniel Rodríguez Rivas — Pre-grasp dinámico desde visión.
#
# Patrón conceptual (consensuado con Fabián, equivalente al pipeline TIAGo+stereo):
#   - El objeto detectado está publicado como TF (o como pose en frame base).
#   - Le pedimos a MoveIt que lleve el LINK 'ur_dual_I_tool0' (flange del UR5)
#     a una pose calculada como: posición_objeto + offset_z_palm + offset_approach
#     con orientación top-down fija (palm mirando hacia abajo).
#   - MoveIt resuelve IK internamente con sus propios planners.
#   - NO usamos compute_ik manual. NO hacemos barrido de orientaciones.
#
# Decisión clave: planificamos para tool0 (no para palm_link). La palma está
# a +0.144 m en Z desde tool0 (montaje físico de la SoftHand). Por eso el
# offset_z final es: object.z + 0.144 (compensación palm) + approach_height.
# -----------------------------------------------------------------------------
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint,
    OrientationConstraint, BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Quaternion
from stereo_location_interfaces.msg import ObjDetArray


def quaternion_from_rpy(roll, pitch, yaw):
    """Roll-Pitch-Yaw (XYZ intrínseco) → quaternion. Sin dependencias externas."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class VisualPregrasp(Node):
    def __init__(self):
        super().__init__("visual_pregrasp")

        # --- Parámetros ---
        self.declare_parameter("planning_group", "Right_arm")
        self.declare_parameter("ee_link", "ur_dual_I_tool0")  # flange del UR5
        self.declare_parameter("base_frame", "ur_dual_I_base_link")
        self.declare_parameter("detections_topic", "/object_tracker/detections_base")
        self.declare_parameter("target_class", "cubo")
        self.declare_parameter("min_confidence", 0.6)
        # Compensación física: distancia desde tool0 hasta la palma de la SoftHand
        self.declare_parameter("palm_z_offset", 0.144)
        # Approach: altura adicional sobre el objeto para el pre-grasp
        self.declare_parameter("approach_height", 0.15)
        # Orientación top-down: tool0 Z apuntando hacia -world Z
        # Para un UR5 con base en el suelo, esto suele ser roll=pi, pitch=0, yaw=0
        # (que orienta el flange con el eje Z apuntando hacia abajo)
        self.declare_parameter("orientation_roll", math.pi)
        self.declare_parameter("orientation_pitch", 0.0)
        self.declare_parameter("orientation_yaw", 0.0)
        # Seguridad
        self.declare_parameter("velocity_scaling", 0.05)
        self.declare_parameter("accel_scaling", 0.05)
        self.declare_parameter("plan_only", True)  # ¡EMPEZAMOS EN PLAN-ONLY!

        # --- Cliente MoveIt ---
        self.move_group_client = ActionClient(self, MoveGroup, "move_action")
        self.get_logger().info("Esperando /move_action...")
        self.move_group_client.wait_for_server()
        self.get_logger().info("MoveGroup OK.")

        # --- Suscriptor detecciones ---
        self.sub = self.create_subscription(
            ObjDetArray,
            self.get_parameter("detections_topic").value,
            self.on_detections,
            10,
        )

        self.last_detection_pos = None  # para evitar disparos repetidos
        self.executing = False

        self.get_logger().info("visual_pregrasp listo. Esperando detecciones...")
        self.get_logger().info(
            f"  target_class = '{self.get_parameter('target_class').value}'"
        )
        self.get_logger().info(
            f"  plan_only    = {self.get_parameter('plan_only').value}"
        )

    def on_detections(self, msg: ObjDetArray):
        if self.executing:
            return

        target_class = self.get_parameter("target_class").value
        min_conf = float(self.get_parameter("min_confidence").value)

        # Filtra por clase y confianza
        candidates = [
            d for d in msg.objects
            if d.class_name.strip().lower() == target_class.strip().lower()
            and d.confidence >= min_conf
        ]
        if not candidates:
            return

        # Toma el de mayor confianza
        target = max(candidates, key=lambda d: d.confidence)

        self.get_logger().info(
            f"Target {target.class_name} @ ({target.position.x:.3f}, "
            f"{target.position.y:.3f}, {target.position.z:.3f}) "
            f"conf={target.confidence:.2f}"
        )

        # Dispara UN ciclo (luego no procesa más hasta acabar)
        self.executing = True
        try:
            self.execute_pregrasp(target.position)
        finally:
            self.executing = False
            self.get_logger().info("Ciclo terminado. Para repetir, relanza el nodo.")
            # Si quieres comportamiento continuo: comenta la línea de abajo
            self.destroy_node()
            rclpy.shutdown()

    def execute_pregrasp(self, object_position):
        """Construye goal y manda a MoveIt."""
        palm_off = float(self.get_parameter("palm_z_offset").value)
        approach = float(self.get_parameter("approach_height").value)
        base_frame = self.get_parameter("base_frame").value
        ee_link = self.get_parameter("ee_link").value
        group = self.get_parameter("planning_group").value
        plan_only = bool(self.get_parameter("plan_only").value)
        vel = float(self.get_parameter("velocity_scaling").value)
        acc = float(self.get_parameter("accel_scaling").value)

        # Pose target: tool0 debe quedar a (object.z + palm_off + approach) en Z
        target_pose = Pose()
        target_pose.position.x = object_position.x
        target_pose.position.y = object_position.y
        target_pose.position.z = object_position.z + palm_off + approach
        target_pose.orientation = quaternion_from_rpy(
            float(self.get_parameter("orientation_roll").value),
            float(self.get_parameter("orientation_pitch").value),
            float(self.get_parameter("orientation_yaw").value),
        )

        self.get_logger().info(
            f"Target tool0 pose: "
            f"pos=({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, "
            f"{target_pose.position.z:.3f}), "
            f"quat=({target_pose.orientation.x:.3f}, {target_pose.orientation.y:.3f}, "
            f"{target_pose.orientation.z:.3f}, {target_pose.orientation.w:.3f})"
        )

        # --- Constraint de posición ---
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = base_frame
        pos_constraint.link_name = ee_link
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0
        # BoundingVolume: una esferita de tolerancia 1 cm
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]  # radio 1 cm
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(target_pose)
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0

        # --- Constraint de orientación ---
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = base_frame
        orient_constraint.link_name = ee_link
        orient_constraint.orientation = target_pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.1   # ~5.7°
        orient_constraint.absolute_y_axis_tolerance = 0.1
        orient_constraint.absolute_z_axis_tolerance = 0.1
        orient_constraint.weight = 1.0

        # --- Goal Constraints ---
        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(orient_constraint)

        # --- MotionPlanRequest ---
        request = MotionPlanRequest()
        request.group_name = group
        request.pipeline_id = "ompl"
        request.planner_id = "RRTstar"
        request.max_velocity_scaling_factor = vel
        request.max_acceleration_scaling_factor = acc
        request.allowed_planning_time = 20.0
        request.num_planning_attempts = 20
        request.goal_constraints.append(constraints)

        # --- Goal de MoveGroup ---
        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options.plan_only = plan_only
        goal_msg.request.start_state.is_diff = True
        goal_msg.planning_options.planning_scene_diff.is_diff = True
        goal_msg.planning_options.planning_scene_diff.robot_state.is_diff = True

        self.get_logger().info(
            f"Enviando goal (plan_only={plan_only}, vel={vel}, acc={acc})..."
        )
        send_future = self.move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)
        gh = send_future.result()
        if not gh.accepted:
            self.get_logger().error("Goal RECHAZADO por MoveIt.")
            return

        result_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result.result.error_code.val == 1:
            self.get_logger().info("✓ Plan/ejecución OK.")
        else:
            self.get_logger().error(
                f"✗ Falló. error_code = {result.result.error_code.val}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = VisualPregrasp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
