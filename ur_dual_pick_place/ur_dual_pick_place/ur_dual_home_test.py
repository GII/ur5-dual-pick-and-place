#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# ur_dual_home_test.py
#
# Prueba mínima de la nueva arquitectura:
#   1. Inicializa UrDualMoveItPy.
#   2. Planifica el brazo derecho lógico hacia Home_Right.
#   3. Deja revisar en RViz.
#   4. Pregunta si se desea ejecutar.
#
# No usa YOLO todavía.
# No usa pose cartesiana todavía.
# -----------------------------------------------------------------------------

import rclpy

from ur_dual_pick_place.ur_dual_moveit_py import UrDualMoveItPy


def main():
    rclpy.init()

    node = UrDualMoveItPy(name="ur_dual_commander")

    node.get_logger().info("Prueba Home_Right iniciada.")

    success, status, plan_result = node.arm_go_to_named_pose(
        arm="right",
        pose_name="Home_Right",
        execute=False,
        velocity_scaling=0.05,
        acceleration_scaling=0.05,
    )

    if not success or plan_result is None:
        node.get_logger().error(f"No se pudo planificar Home_Right. Status: {status}")
        rclpy.shutdown()
        return

    node.get_logger().info("═" * 70)
    node.get_logger().info("Plan hacia Home_Right generado.")
    node.get_logger().info("Revisa RViz. Si el plan se ve seguro, escribe 'yes'.")
    node.get_logger().info("Cualquier otra respuesta cancela la ejecución.")
    node.get_logger().info("═" * 70)

    try:
        answer = input("> ").strip().lower()
    except EOFError:
        answer = "no"

    if answer == "yes":
        node.execute_trajectory(
            plan_result.trajectory,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )
    else:
        node.get_logger().info("Ejecución cancelada. Solo se planificó.")

    node.get_logger().info("Nodo vivo. Ctrl+C para cerrar.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
