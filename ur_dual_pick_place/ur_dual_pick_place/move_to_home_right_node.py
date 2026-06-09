#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# TFG Daniel Rodríguez Rivas
# Hito 1: mover Right_arm a Home_Right usando el patrón de Fabián:
# ActionClient sobre /move_action + MotionPlanRequest + JointConstraint.
#
# IMPORTANTE:
# Los valores de Home_Right vienen del SRDF y ya están en radianes.
# Por eso NO se usa math.radians().
# -----------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint


GROUP_NAME = "Right_arm"

VELOCITY_SCALING = 0.10
ACCELERATION_SCALING = 0.10

HOME_RIGHT_JOINTS_RAD = {
    "ur_dual_I_elbow_joint": 0.0,
    "ur_dual_I_shoulder_lift_joint": -1.5708,
    "ur_dual_I_shoulder_pan_joint": 0.0,
    "ur_dual_I_wrist_1_joint": -1.57075,
    "ur_dual_I_wrist_2_joint": 0.0,
    "ur_dual_I_wrist_3_joint": 0.0,
}


class MoveToHomeRight(Node):
    def __init__(self):
        super().__init__("move_to_home_right")

        self.move_group_client = ActionClient(self, MoveGroup, "move_action")

        self.get_logger().info("Esperando action server /move_action...")
        while not self.move_group_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Todavía esperando /move_action...")

        self.get_logger().info("/move_action conectado.")

    def create_joint_goal_request(self, joint_values_rad):
        request = MotionPlanRequest()
        request.group_name = GROUP_NAME
        request.max_velocity_scaling_factor = VELOCITY_SCALING
        request.max_acceleration_scaling_factor = ACCELERATION_SCALING
        request.allowed_planning_time = 10.0
        request.num_planning_attempts = 5

        constraints = Constraints()

        for joint_name, joint_value_rad in joint_values_rad.items():
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(joint_value_rad)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        request.goal_constraints.append(constraints)
        return request

    def go_home(self):
        goal_msg = MoveGroup.Goal()
        goal_msg.request = self.create_joint_goal_request(HOME_RIGHT_JOINTS_RAD)

        goal_msg.planning_options.planning_scene_diff.is_diff = True
        goal_msg.planning_options.planning_scene_diff.robot_state.is_diff = True

        self.get_logger().info(
            f"Enviando Home_Right para grupo {GROUP_NAME} con "
            f"velocidad {int(VELOCITY_SCALING * 100)}%..."
        )

        send_future = self.move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if goal_handle is None:
            self.get_logger().error("No se recibió goal_handle desde MoveIt.")
            return False

        if not goal_handle.accepted:
            self.get_logger().error("Goal rechazado por MoveIt.")
            return False

        self.get_logger().info("Goal aceptado. Esperando resultado...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        if result is None:
            self.get_logger().error("No se recibió resultado desde MoveIt.")
            return False

        error_code = result.result.error_code.val

        if error_code == 1:
            self.get_logger().info("✓ Right_arm llegó a Home_Right correctamente.")
            return True

        self.get_logger().error(f"✗ Movimiento falló. MoveIt error_code: {error_code}")
        return False


def main(args=None):
    rclpy.init(args=args)
    node = MoveToHomeRight()

    try:
        node.go_home()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
