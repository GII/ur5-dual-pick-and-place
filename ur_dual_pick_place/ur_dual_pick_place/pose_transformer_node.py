#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# TFG "Sistema de manipulación 3D dual-arm con UR5+SoftHand"
# Daniel Rodríguez Rivas, GII/UDC, 2026.
#
# Etapa 1b:
# Transforma detecciones 3D desde el frame óptico de la OAK-D
# hacia el frame base del brazo derecho del sistema dual UR5.
# -----------------------------------------------------------------------------

import math
from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point

from geometry_msgs.msg import PointStamped
from stereo_location_interfaces.msg import ObjDetArray, ObjDet


class PoseTransformer(Node):
    """
    Entrada:
        /object_tracker/detections
        ObjDetArray en oak_rgb_camera_optical_frame

    Salida:
        /object_tracker/detections_base
        ObjDetArray en ur_dual_I_base_link

    Decisión importante:
        Se usa msg.header.frame_id como frame fuente.
        NO se usa det.header.frame_id, porque ese campo contiene frames
        individuales tipo detection_cubo_1.
    """

    def __init__(self):
        super().__init__('pose_transformer')

        self.declare_parameter('input_topic', '/object_tracker/detections')
        self.declare_parameter('output_topic', '/object_tracker/detections_base')
        self.declare_parameter('target_frame', 'ur_dual_I_base_link')
        self.declare_parameter('tf_timeout_s', 0.2)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.tf_timeout = Duration(
            seconds=float(self.get_parameter('tf_timeout_s').value)
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            ObjDetArray,
            self.input_topic,
            self.on_detections,
            10
        )

        self.pub = self.create_publisher(
            ObjDetArray,
            self.output_topic,
            10
        )

        self.get_logger().info(
            f'pose_transformer listo | '
            f'In: {self.input_topic} | '
            f'Out: {self.output_topic} | '
            f'Target frame: {self.target_frame}'
        )

    @staticmethod
    def _is_finite_point(point):
        return (
            math.isfinite(point.x) and
            math.isfinite(point.y) and
            math.isfinite(point.z)
        )

    def on_detections(self, msg: ObjDetArray):
        out = ObjDetArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.target_frame

        if not msg.objects:
            self.pub.publish(out)
            return

        source_frame = msg.header.frame_id

        if not source_frame:
            self.get_logger().warn(
                'ObjDetArray recibido sin header.frame_id; no se puede transformar.',
                throttle_duration_sec=2.0
            )
            return

        try:
            # Primero intenta transformar usando el timestamp del mensaje.
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                msg.header.stamp,
                timeout=self.tf_timeout
            )
        except TransformException as exact_error:
            try:
                # Fallback: usar el TF más reciente disponible.
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    Time(),
                    timeout=self.tf_timeout
                )
                self.get_logger().warn(
                    f'TF con timestamp exacto falló; usando latest. Detalle: {exact_error}',
                    throttle_duration_sec=2.0
                )
            except TransformException as latest_error:
                self.get_logger().warn(
                    f'TF lookup falló de {source_frame} a {self.target_frame}: {latest_error}',
                    throttle_duration_sec=2.0
                )
                return

        for det in msg.objects:
            if not self._is_finite_point(det.position):
                self.get_logger().warn(
                    f'Skipping {det.class_name}: posición inválida '
                    f'({det.position.x}, {det.position.y}, {det.position.z})',
                    throttle_duration_sec=2.0
                )
                continue

            ps = PointStamped()
            ps.header.stamp = msg.header.stamp
            ps.header.frame_id = source_frame
            ps.point = det.position

            try:
                ps_t = do_transform_point(ps, tf)
            except Exception as e:
                self.get_logger().warn(
                    f'Skipping {det.class_name}: error transformando punto ({e})',
                    throttle_duration_sec=2.0
                )
                continue

            new_det = ObjDet()
            new_det.header.stamp = msg.header.stamp
            new_det.header.frame_id = self.target_frame
            new_det.class_name = det.class_name
            new_det.confidence = det.confidence
            new_det.position = ps_t.point
            new_det.bounding_box = deepcopy(det.bounding_box)
            new_det.dimensions = deepcopy(det.dimensions)

            out.objects.append(new_det)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PoseTransformer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
