#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# obstacle_clusterer.py
#
# Detecta obstáculos automáticamente desde la nube de puntos de la OAK-D y los
# publica como CollisionObjects en el planning scene de MoveIt.
#
# Activación: servicio /ur_dual/update_obstacles (Trigger).
# El commander llama a este servicio ANTES de cada planificación.
#
# Pipeline:
#   1. Capturar el último frame depth + camera_info.
#   2. Reconstruir nube en frame cámara.
#   3. TF lookup propio (Time() = más reciente) y transformar a base.
#   4. Recortar al workspace conocido.
#   5. RANSAC plano (Open3D) para quitar la mesa.
#   6. Excluir esfera alrededor de la pose YOLO del objeto target (si existe).
#   7. DBSCAN (sklearn) para clusterizar lo restante.
#   8. Por cada cluster válido: bounding box AABB.
#   9. Publicar como CollisionObjects (operación ADD; los previos se REMOVE).
# -----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)],
    ], dtype=np.float64)


class ObstacleClusterer(Node):
    OBSTACLE_ID_PREFIX = "auto_obstacle_"

    def __init__(self):
        super().__init__("obstacle_clusterer")

        # ---- Parámetros configurables ----
        self.declare_parameter("depth_topic", "/oak_cam/oak/stereo/image_raw")
        self.declare_parameter("camera_info_topic", "/oak_cam/oak/stereo/camera_info")
        self.declare_parameter("base_frame", "ur_dual_I_base_link")
        self.declare_parameter("object_pose_topic", "/ur_dual/object_pose")

        # Workspace bounds en frame base
        self.declare_parameter("min_x", -0.75)
        self.declare_parameter("max_x", 0.25)
        self.declare_parameter("min_y", 0.30)
        self.declare_parameter("max_y", 1.10)
        self.declare_parameter("min_z", 0.005)
        self.declare_parameter("max_z", 0.60)

        # Filtro profundidad cámara
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 1.50)
        self.declare_parameter("depth_subsample", 3)

        # RANSAC mesa
        self.declare_parameter("ransac_distance_threshold", 0.012)
        self.declare_parameter("ransac_n_iterations", 500)
        self.declare_parameter("expected_table_normal_z_min", 0.85)

        # Exclusión objeto target
        self.declare_parameter("object_exclusion_radius", 0.10)
        self.declare_parameter("object_pose_max_age_s", 3.0)

        # DBSCAN
        self.declare_parameter("dbscan_eps", 0.03)
        self.declare_parameter("dbscan_min_samples", 30)
        self.declare_parameter("min_cluster_size", 80)
        self.declare_parameter("max_clusters", 8)

        # Padding de las cajas (cm extra alrededor del bbox real)
        self.declare_parameter("bbox_padding", 0.02)

        # ---- Estado interno ----
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_object_pose = None
        self.latest_object_pose_time_s = 0.0
        self.last_published_ids = set()

        # ---- TF ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Suscriptores ----
        self.create_subscription(Image,
            self.get_parameter("depth_topic").value,
            self._on_depth, 10)
        self.create_subscription(CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._on_camera_info, 10)
        self.create_subscription(PoseStamped,
            self.get_parameter("object_pose_topic").value,
            self._on_object_pose, 10)

        # ---- Publisher al planning scene ----
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)

        # ---- Servicio principal ----
        self.create_service(Trigger,
            "/ur_dual/update_obstacles",
            self._on_update_obstacles)

        # ---- Servicio para limpiar todos los obstáculos auto ----
        self.create_service(Trigger,
            "/ur_dual/clear_auto_obstacles",
            self._on_clear_auto_obstacles)

        self.get_logger().info(
            "ObstacleClusterer listo.\n"
            "  Servicios:\n"
            "    /ur_dual/update_obstacles    -> captura escena y publica obstáculos\n"
            "    /ur_dual/clear_auto_obstacles -> limpia obstáculos auto previos"
        )

    # =========================================================================
    # Callbacks de cache
    # =========================================================================
    def _on_depth(self, msg):
        self.latest_depth = msg

    def _on_camera_info(self, msg):
        self.latest_camera_info = msg

    def _on_object_pose(self, msg):
        self.latest_object_pose = msg
        self.latest_object_pose_time_s = self.get_clock().now().nanoseconds * 1e-9

    # =========================================================================
    # Servicio principal: captura escena y publica obstáculos
    # =========================================================================
    def _on_update_obstacles(self, request, response):
        try:
            n = self._capture_and_publish()
            response.success = True
            response.message = f"Obstáculos detectados y publicados: {n}"
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = f"Error en captura: {exc}"
            self.get_logger().error(response.message)
        return response

    def _on_clear_auto_obstacles(self, request, response):
        n = self._remove_all_auto_obstacles()
        response.success = True
        response.message = f"Removidos {n} obstáculos auto."
        self.get_logger().info(response.message)
        return response

    # =========================================================================
    # Pipeline completo
    # =========================================================================
    def _capture_and_publish(self):
        # --- 1. Validar inputs ---
        if self.latest_depth is None:
            raise RuntimeError("Aún no llegó ningún frame de depth.")
        if self.latest_camera_info is None:
            raise RuntimeError("Aún no llegó camera_info.")

        depth_msg = self.latest_depth
        cam_info = self.latest_camera_info

        if depth_msg.encoding != "16UC1":
            raise RuntimeError(f"Encoding depth no soportado: {depth_msg.encoding}")

        # --- 2. Construir nube en frame cámara ---
        points_cam = self._depth_to_points_cam(depth_msg, cam_info)
        if points_cam.shape[0] == 0:
            self.get_logger().warn("Nube vacía tras filtrar por rango de profundidad.")
            self._remove_all_auto_obstacles()
            return 0

        # --- 3. Transformar a base ---
        base_frame = self.get_parameter("base_frame").value
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                base_frame,
                depth_msg.header.frame_id,
                Time(),  # más reciente, no stamp del depth
                timeout=Duration(seconds=0.5),
            )
        except Exception as exc:
            raise RuntimeError(f"TF {base_frame} <- {depth_msg.header.frame_id} falló: {exc}")

        R = quaternion_to_rotation_matrix(tf_msg.transform.rotation)
        t = np.array([
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
        ], dtype=np.float64)
        points_base = (R @ points_cam.T).T + t

        # --- 4. Recortar al workspace ---
        points_ws = self._crop_workspace(points_base)
        self.get_logger().info(
            f"Tras workspace crop: {points_ws.shape[0]} puntos "
            f"(de {points_base.shape[0]} totales)"
        )
        if points_ws.shape[0] < 200:
            self.get_logger().warn(
                "Pocos puntos en el workspace tras crop. Revisar ROI o pose del robot."
            )
            self._remove_all_auto_obstacles()
            return 0

        # --- 5. RANSAC: quitar mesa ---
        points_no_table = self._remove_table_ransac(points_ws)
        self.get_logger().info(
            f"Tras quitar mesa (RANSAC): {points_no_table.shape[0]} puntos"
        )
        if points_no_table.shape[0] < 50:
            self.get_logger().warn("Pocos puntos tras quitar mesa.")
            self._remove_all_auto_obstacles()
            return 0

        # --- 6. Excluir objeto target ---
        points_no_obj = self._exclude_object_target(points_no_table)
        self.get_logger().info(
            f"Tras excluir objeto target: {points_no_obj.shape[0]} puntos"
        )

        # --- 7. DBSCAN ---
        clusters = self._cluster_dbscan(points_no_obj)
        self.get_logger().info(f"Clusters detectados: {len(clusters)}")

        # --- 8. Bounding boxes ---
        bboxes = [self._aabb_with_padding(c) for c in clusters]

        # --- 9. Publicar al planning scene ---
        self._remove_all_auto_obstacles()  # limpiar previos
        self._publish_bboxes(bboxes, base_frame)

        return len(bboxes)

    # =========================================================================
    # Etapas del pipeline
    # =========================================================================
    def _depth_to_points_cam(self, depth_msg, cam_info):
        h, w = depth_msg.height, depth_msg.width
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape((h, w))

        sub = int(self.get_parameter("depth_subsample").value)
        min_d = float(self.get_parameter("min_depth_m").value)
        max_d = float(self.get_parameter("max_depth_m").value)

        k = cam_info.k
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if fx <= 0 or fy <= 0:
            raise RuntimeError("CameraInfo inválido (fx/fy <= 0).")

        vs = np.arange(0, h, sub)
        us = np.arange(0, w, sub)
        uu, vv = np.meshgrid(us, vs)
        d = depth[vv, uu].astype(np.float32) / 1000.0

        valid = (d > min_d) & (d < max_d) & np.isfinite(d)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64)

        z = d[valid]
        x = (uu[valid].astype(np.float32) - cx) * z / fx
        y = (vv[valid].astype(np.float32) - cy) * z / fy
        return np.stack([x, y, z], axis=1).astype(np.float64)

    def _crop_workspace(self, points_base):
        min_x = float(self.get_parameter("min_x").value)
        max_x = float(self.get_parameter("max_x").value)
        min_y = float(self.get_parameter("min_y").value)
        max_y = float(self.get_parameter("max_y").value)
        min_z = float(self.get_parameter("min_z").value)
        max_z = float(self.get_parameter("max_z").value)

        m = ((points_base[:, 0] >= min_x) & (points_base[:, 0] <= max_x)
             & (points_base[:, 1] >= min_y) & (points_base[:, 1] <= max_y)
             & (points_base[:, 2] >= min_z) & (points_base[:, 2] <= max_z))
        return points_base[m]

    def _remove_table_ransac(self, points_base):
        # Usamos Open3D para RANSAC eficiente.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_base)

        thr = float(self.get_parameter("ransac_distance_threshold").value)
        n_it = int(self.get_parameter("ransac_n_iterations").value)
        z_min = float(self.get_parameter("expected_table_normal_z_min").value)

        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=thr,
                ransac_n=3,
                num_iterations=n_it,
            )
        except Exception as exc:
            self.get_logger().warn(f"RANSAC falló: {exc}; conservo todos los puntos.")
            return points_base

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / (np.linalg.norm(normal) + 1e-9)

        # Aceptamos el plano como mesa solo si su normal es aproximadamente vertical
        # (la mesa es horizontal en base_link). Si no, devolvemos todos los puntos
        # (puede ser que la mesa no esté visible y RANSAC haya encontrado una pared).
        if abs(normal[2]) < z_min:
            self.get_logger().warn(
                f"Plano RANSAC no parece la mesa (normal_z={normal[2]:.3f} < {z_min})."
                " No quito ningún plano."
            )
            return points_base

        self.get_logger().info(
            f"Plano mesa detectado: normal=({a:.2f},{b:.2f},{c:.2f}), "
            f"d={d:.3f}, inliers={len(inliers)}/{points_base.shape[0]}"
        )

        mask = np.ones(points_base.shape[0], dtype=bool)
        mask[inliers] = False
        return points_base[mask]

    def _exclude_object_target(self, points_base):
        max_age = float(self.get_parameter("object_pose_max_age_s").value)
        radius = float(self.get_parameter("object_exclusion_radius").value)

        if self.latest_object_pose is None:
            return points_base

        now_s = self.get_clock().now().nanoseconds * 1e-9
        age = now_s - self.latest_object_pose_time_s
        if age > max_age:
            self.get_logger().warn(
                f"Pose objeto stale (edad {age:.1f}s > {max_age}s); no excluyo."
            )
            return points_base

        ox = self.latest_object_pose.pose.position.x
        oy = self.latest_object_pose.pose.position.y
        oz = self.latest_object_pose.pose.position.z

        dx = points_base[:, 0] - ox
        dy = points_base[:, 1] - oy
        dz = points_base[:, 2] - oz
        dist_sq = dx*dx + dy*dy + dz*dz
        mask = dist_sq > (radius * radius)

        n_excluded = np.sum(~mask)
        self.get_logger().info(
            f"Excluidos {n_excluded} puntos en esfera r={radius:.2f}m "
            f"alrededor de objeto ({ox:.2f},{oy:.2f},{oz:.2f})"
        )
        return points_base[mask]

    def _cluster_dbscan(self, points_base):
        if points_base.shape[0] == 0:
            return []

        eps = float(self.get_parameter("dbscan_eps").value)
        min_s = int(self.get_parameter("dbscan_min_samples").value)
        min_size = int(self.get_parameter("min_cluster_size").value)
        max_clusters = int(self.get_parameter("max_clusters").value)

        labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(points_base)

        clusters = []
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            if label == -1:
                continue  # ruido
            cluster_points = points_base[labels == label]
            if cluster_points.shape[0] >= min_size:
                clusters.append(cluster_points)

        # Ordenar por tamaño descendente y limitar
        clusters.sort(key=lambda c: c.shape[0], reverse=True)
        clusters = clusters[:max_clusters]

        for i, c in enumerate(clusters):
            self.get_logger().info(
                f"Cluster {i}: {c.shape[0]} puntos, "
                f"centro=({c.mean(0)[0]:.2f},{c.mean(0)[1]:.2f},{c.mean(0)[2]:.2f})"
            )
        return clusters

    def _aabb_with_padding(self, cluster_points):
        pad = float(self.get_parameter("bbox_padding").value)
        mins = cluster_points.min(axis=0) - pad
        maxs = cluster_points.max(axis=0) + pad
        center = (mins + maxs) / 2.0
        size = maxs - mins
        return center, size

    # =========================================================================
    # Publicación al planning scene
    # =========================================================================
    def _publish_bboxes(self, bboxes, frame_id):
        scene_msg = PlanningScene()
        scene_msg.is_diff = True

        new_ids = set()
        for i, (center, size) in enumerate(bboxes):
            obj_id = f"{self.OBSTACLE_ID_PREFIX}{i}"
            new_ids.add(obj_id)

            collision = CollisionObject()
            collision.header.frame_id = frame_id
            collision.header.stamp = self.get_clock().now().to_msg()
            collision.id = obj_id
            collision.operation = CollisionObject.ADD

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [float(size[0]), float(size[1]), float(size[2])]

            pose = Pose()
            pose.position.x = float(center[0])
            pose.position.y = float(center[1])
            pose.position.z = float(center[2])
            pose.orientation.w = 1.0

            collision.primitives.append(box)
            collision.primitive_poses.append(pose)

            scene_msg.world.collision_objects.append(collision)

        self.scene_pub.publish(scene_msg)
        self.last_published_ids = new_ids

    def _remove_all_auto_obstacles(self):
        if not self.last_published_ids:
            return 0

        scene_msg = PlanningScene()
        scene_msg.is_diff = True

        for obj_id in self.last_published_ids:
            collision = CollisionObject()
            collision.id = obj_id
            collision.operation = CollisionObject.REMOVE
            scene_msg.world.collision_objects.append(collision)

        self.scene_pub.publish(scene_msg)
        n = len(self.last_published_ids)
        self.last_published_ids = set()
        return n


def main():
    rclpy.init()
    node = ObstacleClusterer()
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
