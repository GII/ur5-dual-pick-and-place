#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener


class NormalGraspTemplateCalibrator(Node):
    def __init__(self):
        super().__init__("normal_grasp_template_calibrator")

        self.base_frame = "ur_dual_I_base_link"
        self.palm_link = "qbhand2m1_palm_link"
        self.object_topic = "/ur_dual/object_pose"

        self.latest_object_pose = None
        self.done = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            PoseStamped,
            self.object_topic,
            self._on_object_pose,
            10,
        )

        self.get_logger().info(
            "Calibrador listo.\n"
            "1) Coloca la mano EXACTAMENTE en el pre-grasp correcto.\n"
            "2) Asegúrate de que /ur_dual/object_pose esté publicando la vaca.\n"
            "3) Este script imprimirá los parámetros normal_grasp_*."
        )

    def _on_object_pose(self, msg: PoseStamped):
        if msg.header.frame_id != self.base_frame:
            self.get_logger().warn(
                f"Objeto recibido en frame '{msg.header.frame_id}', "
                f"pero se esperaba '{self.base_frame}'. Se ignora."
            )
            return

        self.latest_object_pose = msg

    def try_compute(self):
        if self.latest_object_pose is None:
            return False

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.palm_link,
                Time(),
                timeout=Duration(seconds=0.5),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"No pude leer TF {self.base_frame} <- {self.palm_link}: {exc}"
            )
            return False

        obj = self.latest_object_pose.pose.position
        palm = transform.transform.translation
        q = transform.transform.rotation

        dx = palm.x - obj.x
        dy = palm.y - obj.y
        dz = palm.z - obj.z

        print("\n================ NORMAL GRASP TEMPLATE ================\n")
        print("Objeto:")
        print(f"  x={obj.x:.6f}, y={obj.y:.6f}, z={obj.z:.6f}")
        print("Palm link:")
        print(f"  x={palm.x:.6f}, y={palm.y:.6f}, z={palm.z:.6f}")
        print("Offset calculado:")
        print(f"  dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
        print("Quaternion palm:")
        print(f"  qx={q.x:.6f}, qy={q.y:.6f}, qz={q.z:.6f}, qw={q.w:.6f}")

        print("\nComandos para setear en el commander:\n")
        print("ros2 param set /ur_dual_commander grasp_mode normal")
        print("ros2 param set /ur_dual_commander pregrasp_pose_link qbhand2m1_palm_link")
        print("ros2 param set /ur_dual_commander pregrasp_orientation_mode grasp_template")
        print(f"ros2 param set /ur_dual_commander normal_grasp_dx {dx:.6f}")
        print(f"ros2 param set /ur_dual_commander normal_grasp_dy {dy:.6f}")
        print(f"ros2 param set /ur_dual_commander normal_grasp_dz {dz:.6f}")
        print(f"ros2 param set /ur_dual_commander normal_grasp_qx {q.x:.6f}")
        print(f"ros2 param set /ur_dual_commander normal_grasp_qy {q.y:.6f}")
        print(f"ros2 param set /ur_dual_commander normal_grasp_qz {q.z:.6f}")
        print(f"ros2 param set /ur_dual_commander normal_grasp_qw {q.w:.6f}")
        print("\n=======================================================\n")

        self.done = True
        return True


def main():
    rclpy.init()
    node = NormalGraspTemplateCalibrator()

    start = time.monotonic()
    timeout_s = 15.0

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.try_compute():
                break

            if time.monotonic() - start > timeout_s:
                node.get_logger().error(
                    "Timeout: no se pudo calcular la plantilla. "
                    "Revisa que /ur_dual/object_pose esté publicando y que TF exista."
                )
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
