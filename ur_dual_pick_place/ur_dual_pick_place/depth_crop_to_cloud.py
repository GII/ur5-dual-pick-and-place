#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
import sensor_msgs_py.point_cloud2 as pc2


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


class DepthCropToCloud(Node):
    def __init__(self):
        super().__init__("depth_crop_to_cloud")

        self.declare_parameter("depth_topic", "/oak_cam/oak/stereo/image_raw")
        self.declare_parameter("camera_info_topic", "/oak_cam/oak/stereo/camera_info")
        self.declare_parameter("output_topic", "/ur_dual/cropped_obstacle_cloud")
        self.declare_parameter("base_frame", "ur_dual_I_base_link")

        # Submuestreo. 6 es liviano. Si queda muy pobre, bajar a 4.
        self.declare_parameter("subsample", 6)

        # Filtro por profundidad desde cámara.
        self.declare_parameter("min_depth_m", 0.45)
        self.declare_parameter("max_depth_m", 1.25)

        # Filtro en coordenadas del robot.
        # Estos rangos son el "volumen útil" donde queremos obstáculos.
        # Ajustables por ros2 param set sin recompilar.
        self.declare_parameter("min_x", -0.75)
        self.declare_parameter("max_x", 0.25)
        self.declare_parameter("min_y", 0.35)
        self.declare_parameter("max_y", 1.10)

        # Muy importante:
        # z_min elimina mesa/floor.
        # Si lo ponés muy alto, deja de ver obstáculos bajos.
        self.declare_parameter("min_z", 0.10)
        self.declare_parameter("max_z", 0.75)

        # Publicar como máximo a esta frecuencia.
        self.declare_parameter("publish_period_s", 1.0)
        self.declare_parameter("min_points_before_publish", 50)

        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.base_frame = self.get_parameter("base_frame").value

        self.latest_camera_info = None
        self.last_publish_time = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cloud_pub = self.create_publisher(PointCloud2, self.output_topic, 10)

        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, 10)
        self.create_subscription(Image, self.depth_topic, self._on_depth, 10)

        self.get_logger().info(
            "DepthCropToCloud listo:\n"
            f"  depth_topic: {self.depth_topic}\n"
            f"  camera_info_topic: {self.camera_info_topic}\n"
            f"  output_topic: {self.output_topic}\n"
            f"  base_frame: {self.base_frame}"
        )

    def _on_camera_info(self, msg: CameraInfo):
        self.latest_camera_info = msg

    def _on_depth(self, msg: Image):
        now = self.get_clock().now().nanoseconds * 1e-9
        publish_period_s = float(self.get_parameter("publish_period_s").value)

        if now - self.last_publish_time < publish_period_s:
            return

        self.last_publish_time = now

        if self.latest_camera_info is None:
            self.get_logger().warn("Aún no hay CameraInfo. No publico nube.")
            return

        if msg.encoding != "16UC1":
            self.get_logger().warn(
                f"Encoding de depth no esperado: {msg.encoding}. Se esperaba 16UC1."
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                msg.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.25),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"No pude transformar {self.base_frame} <- {msg.header.frame_id}: {exc}"
            )
            return

        h = msg.height
        w = msg.width

        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))

        subsample = int(self.get_parameter("subsample").value)
        min_depth_m = float(self.get_parameter("min_depth_m").value)
        max_depth_m = float(self.get_parameter("max_depth_m").value)

        min_x = float(self.get_parameter("min_x").value)
        max_x = float(self.get_parameter("max_x").value)
        min_y = float(self.get_parameter("min_y").value)
        max_y = float(self.get_parameter("max_y").value)
        min_z = float(self.get_parameter("min_z").value)
        max_z = float(self.get_parameter("max_z").value)

        # Intrínsecos.
        k = self.latest_camera_info.k
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])

        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warn("CameraInfo inválido: fx/fy <= 0.")
            return

        v_coords = np.arange(0, h, subsample)
        u_coords = np.arange(0, w, subsample)
        uu, vv = np.meshgrid(u_coords, v_coords)

        d = depth[vv, uu].astype(np.float32) / 1000.0

        valid = np.isfinite(d)
        valid &= d > min_depth_m
        valid &= d < max_depth_m

        if not np.any(valid):
            self._publish_empty(msg.header.stamp)
            return

        z_cam = d[valid]
        x_cam = (uu[valid].astype(np.float32) - cx) * z_cam / fx
        y_cam = (vv[valid].astype(np.float32) - cy) * z_cam / fy

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float64)

        rotation = quaternion_to_rotation_matrix(transform.transform.rotation)
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )

        points_base = (rotation @ points_cam.T).T + translation

        x = points_base[:, 0]
        y = points_base[:, 1]
        z = points_base[:, 2]

        workspace_mask = (
            (x >= min_x)
            & (x <= max_x)
            & (y >= min_y)
            & (y <= max_y)
            & (z >= min_z)
            & (z <= max_z)
        )

        # Filtramos en coordenadas base SOLO para decidir qué puntos conservar,
        # pero publicamos la nube en el frame del sensor (sin transformar puntos).
        # Esto es crítico: el PointCloudOctomapUpdater necesita el frame del
        # sensor para hacer ray-casting clearing (liberar voxels en el camino
        # entre cámara y punto observado). Si publicáramos en base_link, el
        # updater no conoce el origen del rayo y los voxels marcados nunca
        # se borran -> aparecen "bloques fantasma" persistentes.
        points_base = (rotation @ points_cam.T).T + translation

        x = points_base[:, 0]
        y = points_base[:, 1]
        z = points_base[:, 2]

        workspace_mask = (
            (x >= min_x)
            & (x <= max_x)
            & (y >= min_y)
            & (y <= max_y)
            & (z >= min_z)
            & (z <= max_z)
        )

        points_filtered_cam = points_cam[workspace_mask]
        points_filtered_base = points_base[workspace_mask]

        min_points = int(self.get_parameter("min_points_before_publish").value)

        if points_filtered_cam.shape[0] < min_points:
            self._publish_empty(msg.header.stamp, msg.header.frame_id)
            self.get_logger().info(
                f"Nube filtrada descartada por pocos puntos: "
                f"{points_filtered_cam.shape[0]} < {min_points}."
            )
            return

        # Log de bbox en base, útil para verificar que el ROI cae donde debe.
        mins = points_filtered_base.min(axis=0)
        maxs = points_filtered_base.max(axis=0)
        centroid = points_filtered_base.mean(axis=0)

        self.get_logger().info(
            f"BBox(base): x=[{mins[0]:.3f},{maxs[0]:.3f}], "
            f"y=[{mins[1]:.3f},{maxs[1]:.3f}], "
            f"z=[{mins[2]:.3f},{maxs[2]:.3f}], "
            f"centro=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f}), "
            f"puntos={points_filtered_cam.shape[0]}"
        )

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id  # frame del SENSOR, no base

        cloud = pc2.create_cloud_xyz32(
            header,
            points_filtered_cam.astype(np.float32).tolist(),
        )

        self.cloud_pub.publish(cloud)

        self.get_logger().info(
            f"Nube publicada en frame sensor '{msg.header.frame_id}': "
            f"raw_valid={points_cam.shape[0]}, "
            f"filtered={points_filtered_cam.shape[0]}, "
            f"ROI base x=[{min_x:.2f},{max_x:.2f}], "
            f"y=[{min_y:.2f},{max_y:.2f}], z=[{min_z:.2f},{max_z:.2f}]"
        )

    def _publish_empty(self, stamp, frame_id):
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        cloud = pc2.create_cloud_xyz32(header, [])
        self.cloud_pub.publish(cloud)


def main():
    rclpy.init()
    node = DepthCropToCloud()

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
