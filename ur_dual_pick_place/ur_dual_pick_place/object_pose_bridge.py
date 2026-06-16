#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# object_pose_bridge.py
#
# Puente entre stereo_location y ur_dual_pick_place.
#
# Entrada:
#   /object_tracker/detections
#   stereo_location_interfaces/msg/ObjDetArray
#   frame: oak_rgb_camera_optical_frame
#
# Salida:
#   /ur_dual/object_pose
#   geometry_msgs/msg/PoseStamped
#   frame: ur_dual_I_base_link
#
# Selecciona el objeto con mayor confianza, opcionalmente filtrando por clase.
# -----------------------------------------------------------------------------

import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

try:
    from tf2_geometry_msgs import do_transform_pose_stamped
except ImportError:
    # Fallback por si la instalación usa el nombre alternativo.
    from tf2_geometry_msgs import do_transform_pose as do_transform_pose_stamped

from stereo_location_interfaces.msg import ObjDetArray


class ObjectPoseBridge(Node):
    def __init__(self):
        super().__init__("object_pose_bridge")

        self.declare_parameter("input_topic", "/object_tracker/detections")
        self.declare_parameter("output_topic", "/ur_dual/object_pose")
        self.declare_parameter("target_frame", "ur_dual_I_base_link")

        # Si queda vacío, toma el objeto de mayor confianza.
        self.declare_parameter("target_class", "")

        self.declare_parameter("min_confidence", 0.50)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.target_class = self.get_parameter("target_class").value
        self.min_confidence = float(self.get_parameter("min_confidence").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(PoseStamped, self.output_topic, 10)

        self.class_pub = self.create_publisher(
            String,
            "/ur_dual/object_class",
            10,
        )


        self.sub = self.create_subscription(
            ObjDetArray,
            self.input_topic,
            self._on_detections,
            10,
        )

        self.last_log_time = 0.0

        self.get_logger().info(
            "ObjectPoseBridge listo:\n"
            f"  input_topic: {self.input_topic}\n"
            f"  output_topic: {self.output_topic}\n"
            f"  target_frame: {self.target_frame}\n"
            f"  target_class: '{self.target_class}'\n"
            f"  min_confidence: {self.min_confidence:.2f}"
        )

    def _get_detection_list(self, msg: ObjDetArray):
        """Soporta repos donde el campo se llama objects o detections."""

        if hasattr(msg, "objects"):
            return list(msg.objects)

        if hasattr(msg, "detections"):
            return list(msg.detections)

        self.get_logger().error(
            "ObjDetArray no tiene campo 'objects' ni 'detections'. "
            "Revisa: ros2 interface show stereo_location_interfaces/msg/ObjDetArray"
        )
        return []

    def _select_best_detection(self, detections):
        """Selecciona la detección de mayor confianza, filtrando clase si aplica."""

        candidates = []

        for det in detections:
            class_name = getattr(det, "class_name", "")
            confidence = float(getattr(det, "confidence", 0.0))

            if confidence < self.min_confidence:
                continue

            if self.target_class and class_name != self.target_class:
                continue

            candidates.append(det)

        if not candidates:
            return None

        return max(candidates, key=lambda d: float(getattr(d, "confidence", 0.0)))

    def _on_detections(self, msg: ObjDetArray):
        detections = self._get_detection_list(msg)

        if not detections:
            return

        best = self._select_best_detection(detections)

        if best is None:
            now = time.monotonic()
            if now - self.last_log_time > 2.0:
                self.get_logger().warn(
                    f"No hay detecciones válidas para target_class='{self.target_class}' "
                    f"con min_confidence={self.min_confidence:.2f}."
                )
                self.last_log_time = now
            return

        source_frame = msg.header.frame_id

        if not source_frame:
            self.get_logger().warn("El ObjDetArray llegó sin header.frame_id. Se ignora.")
            return

        pose_camera = PoseStamped()
        pose_camera.header.stamp = msg.header.stamp
        pose_camera.header.frame_id = source_frame
        pose_camera.pose.position = best.position
        pose_camera.pose.orientation.w = 1.0

        try:
            # Para una TF estática cámara->base, Time() evita problemas de extrapolación.
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.5),
            )

            pose_base = do_transform_pose_stamped(pose_camera, transform)

        except Exception as exc:
            now = time.monotonic()
            if now - self.last_log_time > 1.0:
                self.get_logger().warn(
                    f"TF falló {self.target_frame} <- {source_frame}: {exc}"
                )
                self.last_log_time = now
            return

        pose_base.header.stamp = self.get_clock().now().to_msg()
        pose_base.header.frame_id = self.target_frame

        self.pub.publish(pose_base)

        class_msg = String()
        class_msg.data = str(best.class_name)
        self.class_pub.publish(class_msg)

        now = time.monotonic()
        if now - self.last_log_time > 1.0:
            self.get_logger().info(
                "Objeto seleccionado y transformado:\n"
                f"  class_name: {best.class_name}\n"
                f"  confidence: {float(best.confidence):.3f}\n"
                f"  camera_frame: {source_frame}\n"
                f"  target_frame: {self.target_frame}\n"
                f"  camera xyz: "
                f"{best.position.x:.3f}, {best.position.y:.3f}, {best.position.z:.3f}\n"
                f"  base xyz: "
                f"{pose_base.pose.position.x:.3f}, "
                f"{pose_base.pose.position.y:.3f}, "
                f"{pose_base.pose.position.z:.3f}"
            )
            self.last_log_time = now


def main():
    rclpy.init()

    node = ObjectPoseBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
