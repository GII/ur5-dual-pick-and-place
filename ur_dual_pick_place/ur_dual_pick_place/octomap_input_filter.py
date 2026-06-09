#!/usr/bin/env python3
"""
octomap_input_filter.py

Filtra la imagen de profundidad del OAK para eliminar los pixeles cuyo
proyectado 3D cae dentro de la AABB del objeto detectado por YOLO. La
imagen filtrada se republica en un topic propio que consume el
DepthImageOctomapUpdater. Como el objeto nunca aparece en los datos
de entrada del updater, tampoco aparece en el OctoMap.

Inspirado en el tutorial MoveIt2 'perception_pipeline/cylinder_segment'.

Topics:
  IN  /oak_cam/oak/stereo/image_raw           (sensor_msgs/Image 16UC1)
  IN  /oak_cam/oak/stereo/camera_info         (sensor_msgs/CameraInfo)
  IN  /ur_dual/object_pose                    (geometry_msgs/PoseStamped)
  IN  /ur_dual/object_class                   (std_msgs/String)
  OUT /oak_cam/oak/stereo/image_raw_filtered  (sensor_msgs/Image 16UC1)
  OUT /oak_cam/oak/stereo/camera_info_filtered(sensor_msgs/CameraInfo)

Servicios:
  /ur_dual/octomap_filter/enable (SetBool)
    - True: filtra el objeto (default)
    - False: pass-through (publica raw)
"""

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker


# Dimensiones aproximadas del objeto SIN margen (size_x, size_y, size_z) en m.
# Mismo dict que el commander. La AABB se construye en frame base con la
# pose del objeto como centro.
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


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


class OctomapInputFilter(Node):

    def __init__(self):
        super().__init__("octomap_input_filter")

        # --- Parámetros ---
        self.declare_parameter("depth_topic_in", "/oak_cam/oak/stereo/image_raw")
        self.declare_parameter("caminfo_topic_in", "/oak_cam/oak/stereo/camera_info")
        self.declare_parameter("depth_topic_out", "/oak_cam/oak/stereo/image_raw_filtered")
        self.declare_parameter("caminfo_topic_out", "/oak_cam/oak/stereo/camera_info_filtered")
        self.declare_parameter("base_frame", "ur_dual_I_base_link")
        self.declare_parameter("object_pose_topic", "/ur_dual/object_pose")
        self.declare_parameter("object_class_topic", "/ur_dual/object_class")
        # Margen adicional alrededor del AABB del objeto (en m, por lado).
        # El pose YOLO no siempre cae exactamente en el centroide del objeto,
        # un margen de 4-5cm cubre errores típicos.
        self.declare_parameter("filter_margin", 0.04)
        # Si la pose es más vieja que esto, se hace pass-through (no se filtra).
        self.declare_parameter("object_pose_max_age_s", 5.0)
        # Valor al que se setean los pixeles filtrados. 0 = invalid (no
        # measurement). El plugin de OctoMap ignora 0, así que esos
        # pixeles no agregan voxels nuevos.
        self.declare_parameter("clear_value_mm", 0)

        self.base_frame = self.get_parameter("base_frame").value
        depth_in = self.get_parameter("depth_topic_in").value
        caminfo_in = self.get_parameter("caminfo_topic_in").value
        depth_out = self.get_parameter("depth_topic_out").value
        caminfo_out = self.get_parameter("caminfo_topic_out").value

        # --- Estado ---
        self.latest_caminfo = None
        self.latest_object_pose = None
        self.latest_object_pose_time = None
        self.latest_object_class = None
        self.filter_enabled = True
        self.n_frames_logged = 0

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- QoS matching de la OAK (BEST_EFFORT no, usa RELIABLE) ---
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE

        # --- Subscriptores ---
        self.create_subscription(Image, depth_in, self._on_depth, qos)
        self.create_subscription(CameraInfo, caminfo_in, self._on_caminfo, qos)
        self.create_subscription(
            PoseStamped,
            self.get_parameter("object_pose_topic").value,
            self._on_object_pose,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("object_class_topic").value,
            self._on_object_class,
            10,
        )

        # --- Publishers ---
        self.depth_pub = self.create_publisher(Image, depth_out, qos)
        self.caminfo_pub = self.create_publisher(CameraInfo, caminfo_out, qos)
        # Marker para visualizar la AABB del objeto target en RViz.
        # Permite confirmar que la caja de filtrado / attach cae donde
        # corresponde respecto al objeto físico.
        self.marker_pub = self.create_publisher(
            Marker,
            "/ur_dual/target_object_marker",
            10,
        )
        # Timer para publicar el marker a una tasa fija (independiente
        # del flujo de depth).
        self.create_timer(0.2, self._publish_marker)




        # --- Servicio enable/disable ---
        self.create_service(
            SetBool,
            "/ur_dual/octomap_filter/enable",
            self._on_enable,
        )

        self.get_logger().info(
            "OctomapInputFilter listo.\n"
            f"  IN  depth:   {depth_in}\n"
            f"  IN  info:    {caminfo_in}\n"
            f"  OUT depth:   {depth_out}\n"
            f"  OUT info:    {caminfo_out}\n"
            f"  base_frame:  {self.base_frame}\n"
            f"  filter:      {'enabled' if self.filter_enabled else 'disabled'}\n"
            "Servicio:\n"
            "  ros2 service call /ur_dual/octomap_filter/enable "
            "std_srvs/srv/SetBool \"{data: true}\""
        )

    # ----- Callbacks de cache -----
    def _on_caminfo(self, msg):
        self.latest_caminfo = msg

    def _on_object_pose(self, msg):
        self.latest_object_pose = msg
        self.latest_object_pose_time = self.get_clock().now()

    def _on_object_class(self, msg):
        self.latest_object_class = msg.data

    def _on_enable(self, request, response):
        self.filter_enabled = bool(request.data)
        response.success = True
        response.message = (
            f"Filtro {'habilitado' if self.filter_enabled else 'deshabilitado'}."
        )
        self.get_logger().info(response.message)
        return response

    # ----- Decisión de filtrar -----
    def _should_filter(self):
        if not self.filter_enabled:
            return False
        if self.latest_object_pose is None:
            return False
        if self.latest_caminfo is None:
            return False
        if self.latest_object_pose_time is None:
            return False
        age = (self.get_clock().now() - self.latest_object_pose_time).nanoseconds * 1e-9
        max_age = float(self.get_parameter("object_pose_max_age_s").value)
        if age > max_age:
            return False
        return True

    # ----- Pipeline principal -----
    def _on_depth(self, msg):
        # Siempre republicamos camera_info con el mismo timestamp del depth,
        # para que el sincronizador del DepthImageOctomapUpdater no descarte
        # pares.
        if self.latest_caminfo is not None:
            ci = CameraInfo()
            ci.header.stamp = msg.header.stamp
            ci.header.frame_id = self.latest_caminfo.header.frame_id
            ci.height = self.latest_caminfo.height
            ci.width = self.latest_caminfo.width
            ci.distortion_model = self.latest_caminfo.distortion_model
            ci.d = self.latest_caminfo.d
            ci.k = self.latest_caminfo.k
            ci.r = self.latest_caminfo.r
            ci.p = self.latest_caminfo.p
            ci.binning_x = self.latest_caminfo.binning_x
            ci.binning_y = self.latest_caminfo.binning_y
            ci.roi = self.latest_caminfo.roi
            self.caminfo_pub.publish(ci)

        # Si no toca filtrar, pass-through directo.
        if not self._should_filter():
            self.depth_pub.publish(msg)
            return

        try:
            filtered = self._filter_depth(msg)
            self.depth_pub.publish(filtered)
        except Exception as exc:
            self.get_logger().warn(
                f"Falló filtrado de depth ({exc}). Pass-through esta vez."
            )
            self.depth_pub.publish(msg)

    def _filter_depth(self, depth_msg):
        if depth_msg.encoding != "16UC1":
            raise RuntimeError(f"Encoding no soportado: {depth_msg.encoding}")

        # Intrinsics
        k = self.latest_caminfo.k
        fx, fy = float(k[0]), float(k[4])
        cx, cy = float(k[2]), float(k[5])
        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError("CameraInfo inválido (fx/fy <= 0).")

        # TF camera -> base
        tf = self.tf_buffer.lookup_transform(
            self.base_frame,
            depth_msg.header.frame_id,
            Time(),
            timeout=Duration(seconds=0.2),
        )
        R = quaternion_to_rotation_matrix(tf.transform.rotation)
        t = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ], dtype=np.float64)

        # AABB del objeto en frame base
        margin = float(self.get_parameter("filter_margin").value)
        obj_class = self.latest_object_class
        if obj_class in OBJECT_DIMENSIONS:
            dx, dy, dz = OBJECT_DIMENSIONS[obj_class]
        else:
            dx, dy, dz = OBJECT_DIM_DEFAULT

        ox = self.latest_object_pose.pose.position.x
        oy = self.latest_object_pose.pose.position.y
        oz = self.latest_object_pose.pose.position.z

        bx_min = ox - dx / 2.0 - margin
        bx_max = ox + dx / 2.0 + margin
        by_min = oy - dy / 2.0 - margin
        by_max = oy + dy / 2.0 + margin
        bz_min = oz - dz / 2.0 - margin
        bz_max = oz + dz / 2.0 + margin

        # Depth como numpy
        h, w = depth_msg.height, depth_msg.width
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape((h, w)).copy()

        # Pixeles válidos (depth > 0)
        z_cam_full = depth.astype(np.float32) / 1000.0
        valid = z_cam_full > 0.10  # ignorar depths muy pequeños / inválidos

        if not np.any(valid):
            return depth_msg  # nada que filtrar, envío original

        # Solo procesar pixeles válidos para ahorrar memoria/tiempo
        rows, cols = np.where(valid)
        z_cam = z_cam_full[rows, cols]
        x_cam = (cols.astype(np.float32) - cx) * z_cam / fx
        y_cam = (rows.astype(np.float32) - cy) * z_cam / fy

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float64)
        pts_base = (R @ pts_cam.T).T + t  # (N, 3)

        in_aabb = (
            (pts_base[:, 0] >= bx_min) & (pts_base[:, 0] <= bx_max)
            & (pts_base[:, 1] >= by_min) & (pts_base[:, 1] <= by_max)
            & (pts_base[:, 2] >= bz_min) & (pts_base[:, 2] <= bz_max)
        )

        clear_value = int(self.get_parameter("clear_value_mm").value)
        rows_clear = rows[in_aabb]
        cols_clear = cols[in_aabb]
        depth[rows_clear, cols_clear] = clear_value

        # Log cada N frames para no saturar
        self.n_frames_logged += 1
        if self.n_frames_logged % 30 == 0:
            self.get_logger().info(
                f"Filtré {len(rows_clear)} pixeles del objeto "
                f"'{obj_class}' en este frame. "
                f"AABB base: x=[{bx_min:.2f},{bx_max:.2f}] "
                f"y=[{by_min:.2f},{by_max:.2f}] z=[{bz_min:.2f},{bz_max:.2f}]"
            )

        # Construir mensaje de salida
        out = Image()
        out.header = depth_msg.header
        out.height = h
        out.width = w
        out.encoding = depth_msg.encoding
        out.is_bigendian = depth_msg.is_bigendian
        out.step = depth_msg.step
        out.data = depth.tobytes()
        return out



    def _publish_marker(self):
        """
        Publica un Marker tipo CUBE en RViz con la AABB del objeto
        detectado (la misma caja que usa el filtro de depth). Si no
        hay objeto fresco, el marker se borra (DELETE).
        """
        # Si no hay pose o está stale, borrar el marker.
        if (self.latest_object_pose is None
                or self.latest_object_pose_time is None):
            self._publish_delete_marker()
            return

        age = (self.get_clock().now() - self.latest_object_pose_time).nanoseconds * 1e-9
        max_age = float(self.get_parameter("object_pose_max_age_s").value)
        if age > max_age:
            self._publish_delete_marker()
            return

        # Mismas dimensiones que usa el filtro de depth.
        margin = float(self.get_parameter("filter_margin").value)
        obj_class = self.latest_object_class
        if obj_class in OBJECT_DIMENSIONS:
            dx, dy, dz = OBJECT_DIMENSIONS[obj_class]
        else:
            dx, dy, dz = OBJECT_DIM_DEFAULT

        sx = dx + 2.0 * margin
        sy = dy + 2.0 * margin
        sz = dz + 2.0 * margin

        marker = Marker()
        marker.header.frame_id = self.latest_object_pose.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_object_aabb"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = self.latest_object_pose.pose.position
        marker.pose.orientation.w = 1.0
        marker.scale.x = sx
        marker.scale.y = sy
        marker.scale.z = sz
        # Color: cyan semi-transparente. Distinto de voxels (verde/amarillo)
        # y obstáculos del OctoMap. Bien visible sin tapar el contenido.
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.35
        # Lifetime corto: si dejamos de publicar el marker se borra solo.
        marker.lifetime.sec = 1

        self.marker_pub.publish(marker)

    def _publish_delete_marker(self):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_object_aabb"
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)




def main():
    rclpy.init()
    node = OctomapInputFilter()
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
