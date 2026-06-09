#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# ur_dual_command_services.py
#
# Nodo comandante estilo TIAGo:
#   - No usa input()
#   - No planifica automáticamente al iniciar
#   - Expone servicios ROS2 para planificar y ejecutar
#
# Servicios:
#   /ur_dual/plan_home_right
#   /ur_dual/execute_last_plan
# -----------------------------------------------------------------------------

import time
import math
import rclpy
from rclpy.executors import MultiThreadedExecutor
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, SetBool
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    AllowedCollisionMatrix,
    AllowedCollisionEntry,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

from ur_dual_pick_place.ur_dual_moveit_py import UrDualMoveItPy


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convierte roll, pitch, yaw a quaternion.

    Convención XYZ intrínseca, suficiente para definir orientaciones fijas
    de pre-grasp.
    """

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy

    return q



# ============================================================
# Dimensiones aproximadas (size_x, size_y, size_z) en metros 
# ============================================================
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


# ============================================================
# Tolerancias del OrientationConstraint del pre-grasp por categoría.
# Cada tupla es (tol_roll, tol_pitch, tol_yaw) en radianes alrededor
# de la orientación de referencia (la del template de pregrasp).
#
# Filosofía:
#   - roll/pitch estrictos (~0.10 rad ≈ 6°) mantienen la palma orientada
#     correctamente (hacia abajo para objetos en mesa, a 90° para botellas).
#   - yaw libre (3.14 rad = ±π) deja que OMPL elija cualquier ángulo de
#     aproximación alrededor del eje vertical, ampliando soluciones IK.
#
# Para peluches blandos (lechuga, tomate) y bolas, las tolerancias son
# más permisivas porque la SoftHand envuelve bien desde cualquier ángulo.
# ============================================================
GRASP_TOLERANCES = {
    "bola":         (0.10, 0.10, 3.14),   
    "botella rosa": (0.10, 0.10, 3.14),   # palma 90° fija, yaw libre
    "caballo":      (0.10, 0.10, 3.14),
    "cubo":         (0.10, 0.10, 3.14),
    "lechuga":      (1.50, 1.50, 3.14),   # peluche, permisivo
    "pina":         (0.10, 0.10, 3.14),
    "prisma":       (0.10, 0.10, 3.14),
    "refresco":     (0.10, 0.10, 3.14),
    "tomate":       (1.50, 1.50, 3.14),   # peluche, permisivo
    "vaca":         (0.10, 0.10, 3.14),
}
GRASP_TOLERANCE_DEFAULT = (0.10, 0.10, 3.14)



TARGET_COLLISION_ID = "target_object"


class UrDualCommandServices(UrDualMoveItPy):
    def __init__(self, name="ur_dual_commander"):
        super().__init__(name)

        self.pending_plan_result = None
        self.latest_joint_state = None
        self.latest_object_pose = None
        self.last_object_pose_log_time = 0.0

        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10,
        )

        self.create_subscription(
            PoseStamped,
            "/ur_dual/object_pose",
            self._on_object_pose,
            10,
        )

        self.create_subscription(
            String,
            "/ur_dual/object_class",
            self._on_object_class,
            10,
        )

        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10,
        )

        self.hand_pub = self.create_publisher(
            JointTrajectory,
            "/qbhand2m1/qbhand2m1_synergies_trajectory_controller/joint_trajectory",
            10,
        )


        # Publisher al planning_scene para attach/detach del objeto agarrado.
        self.planning_scene_pub = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            10,
        )


        self.update_obstacles_client = self.create_client(
            Trigger,
            "/ur_dual/update_obstacles",
            callback_group=self.cb_internal_client,
        )        



        # TF: lo usamos para leer la pose actual del tool0 y generar
        # una prueba cartesiana segura con un pequeño offset.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parámetros de la prueba cartesiana.
        # Default: mover el tool0 -5 cm en Z respecto a su pose actual.
        self.declare_parameter("offset_test_base_frame", "ur_dual_I_base_link")
        self.declare_parameter("offset_test_pose_link", "ur_dual_I_tool0")
        self.declare_parameter("offset_test_dx", 0.0)
        self.declare_parameter("offset_test_dy", 0.0)
        self.declare_parameter("offset_test_dz", -0.1)

        # Parámetros para pre-grasp a partir de pose de objeto.
        self.declare_parameter("pregrasp_base_frame", "ur_dual_I_base_link")
        # Pre-grasp normal para objetos con palma paralela a la mesa.
        # Usamos tool0 como link de planificación y compensamos la distancia
        # física hacia la palma.
        self.declare_parameter("palm_z_offset", 0.144)
        self.declare_parameter("approach_height", 0.25)
        self.declare_parameter("pregrasp_roll", math.pi)
        self.declare_parameter("pregrasp_pitch", 0.0)
        self.declare_parameter("pregrasp_yaw", 0.0)
        self.declare_parameter("cartesian_safe_z", 1.00)
        self.declare_parameter("cartesian_pregrasp_pose_link", "ur_dual_I_tool0")


        # Modo de agarre:
        #   normal -> vaca, cubo, bola, lego, etc.
        #   bottle -> botellas
        self.declare_parameter("grasp_mode", "normal")

        # Plantilla para objetos normales.
        # Para el entregable mínimo usamos centro del objeto en X/Y y solo offset en Z.
        self.declare_parameter("normal_grasp_dx", 0.0)
        self.declare_parameter("normal_grasp_dy", 0.0)
        self.declare_parameter("normal_grasp_dz", 0.040)

        # Orientación normal nueva, no inclinada.
        self.declare_parameter("normal_grasp_qx", -0.003)
        self.declare_parameter("normal_grasp_qy", 0.023)
        self.declare_parameter("normal_grasp_qz", -0.022)
        self.declare_parameter("normal_grasp_qw", 0.999)

        # Plantilla para botellas.
        # X/Y/Z se ajustan después con prueba real sobre botella.
        self.declare_parameter("bottle_grasp_dx", 0.0)
        self.declare_parameter("bottle_grasp_dy", 0.0)
        self.declare_parameter("bottle_grasp_dz", 0.0)

        # Orientación promedio de las dos poses de botella.
        self.declare_parameter("bottle_grasp_qx", -0.008)
        self.declare_parameter("bottle_grasp_qy", 0.680)
        self.declare_parameter("bottle_grasp_qz", 0.028)
        self.declare_parameter("bottle_grasp_qw", 0.733)

        self.declare_parameter("pregrasp_pose_link", "qbhand2m1_palm_link")
        self.declare_parameter("pregrasp_orientation_mode", "grasp_template")

        # ============================================================
        # Attached object: configuración geométrica del objeto agarrado.
        # Por defecto se modela como una caja conservadora. Los nombres
        # asocian con la clase YOLO detectada. Para el TFG usamos un
        # tamaño único por categoría; un bbox dinámico desde la nube
        # sería trabajo futuro.
        # ============================================================
        self.declare_parameter("attached_object_id", "grasped_object")
        self.declare_parameter("attached_object_frame", "qbhand2m1_palm_link")
        # Tamaño de la caja en metros (ancho, profundo, alto).
        # Tamaño conservador que cubre vaca/cubo/botella pequeña.
        self.declare_parameter("attached_object_size_x", 0.08)
        self.declare_parameter("attached_object_size_y", 0.08)
        self.declare_parameter("attached_object_size_z", 0.12)
        # Offset del centro de la caja respecto al frame de la mano.
        # El centro de la mano cerrada con un objeto agarrado típicamente
        # queda un poco delante de palm_link.
        self.declare_parameter("attached_object_offset_x", 0.0)
        self.declare_parameter("attached_object_offset_y", 0.0)
        self.declare_parameter("attached_object_offset_z", -0.05)
        # Margen extra en cada eje al limpiar voxels del octomap alrededor
        # del objeto. Tolera errores de pose YOLO de hasta este valor (m).
        self.declare_parameter("clear_box_margin", 0.02)
        


        self.create_service(
            Trigger,
            "/ur_dual/plan_home_right",
            self._on_plan_home_right,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_offset_test",
            self._on_plan_offset_test,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_pregrasp_cartesian_from_latest_pose",
            self._on_plan_pregrasp_cartesian_from_latest_pose,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_pregrasp_from_latest_pose",
            self._on_plan_pregrasp_from_latest_pose,
        )

        self.create_service(
            SetBool,
            "/ur_dual/execute_last_plan",
            self._on_execute_last_plan,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_ready_right",
            self._on_plan_ready_right,
        )

        self.create_service(
            Trigger,
            "/ur_dual/open_hand",
            self._on_open_hand,
        )

        self.create_service(
            Trigger,
            "/ur_dual/close_hand",
            self._on_close_hand,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_place_normal",
            self._on_plan_place_normal,
        )

        self.create_service(
            Trigger,
            "/ur_dual/plan_place_bottle",
            self._on_plan_place_bottle,
        )

        self.create_service(
            Trigger,
            "/ur_dual/clear_octomap",
            self._on_clear_octomap,
        )



        # ============================================================
        # Servicios para FREEZE / UNFREEZE del OctoMap.
        # Mecanismo: stop/start del driver de la OAK.
        # Cuando la cámara no publica, el DepthImageOctomapUpdater no
        # recibe frames y el OctoMap queda congelado en su último estado.
        # Cuando volvemos a start, la cámara publica de nuevo y OctoMap
        # se actualiza otra vez (manteniendo voxels viejos por persistencia
        # hasta que el ray-casting los limpie).
        # ============================================================
        self.create_service(
            Trigger,
            "/ur_dual/freeze_octomap",
            self._on_freeze_octomap,
        )
        self.create_service(
            Trigger,
            "/ur_dual/unfreeze_octomap",
            self._on_unfreeze_octomap,
        )

        # Clientes a los servicios stop/start de la OAK.
        self.oak_stop_client = self.create_client(
            Trigger,
            "/oak_cam/oak/stop_camera",
            callback_group=self.cb_internal_client,
        )
        self.oak_start_client = self.create_client(
            Trigger,
            "/oak_cam/oak/start_camera",
            callback_group=self.cb_internal_client,
        )



        # ============================================================
        # Servicios para attach / detach del objeto agarrado.
        # attach: llamar JUSTO DESPUÉS de close_hand exitoso.
        # detach: llamar JUSTO ANTES de open_hand en el place.
        # ============================================================
        self.create_service(
            Trigger,
            "/ur_dual/clear_octomap_around_object",
            self._on_clear_octomap_around_object,
        )
        self.create_service(
            Trigger,
            "/ur_dual/attach_grasped_object",
            self._on_attach_grasped_object,
        )
        self.create_service(
            Trigger,
            "/ur_dual/detach_grasped_object",
            self._on_detach_grasped_object,
        )



        # Flag de estado: True si el OctoMap está congelado (cámara parada).
        self._octomap_frozen = False
        # Flag de estado: True si hay un objeto attached al palm_link actualmente.
        self._object_attached = False

        # Última clase del objeto detectado. La publica object_pose_bridge.
        # La usamos para elegir dimensiones en clear_octomap_around_object
        # y tolerancias en el orientation constraint del pregrasp.
        self._latest_object_class = None



        self.get_logger().info(
            "Servicios disponibles:\n"
            "  ros2 service call /ur_dual/plan_home_right std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/plan_offset_test std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/plan_pregrasp_from_latest_pose std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/execute_last_plan std_srvs/srv/SetBool \"{data: true}\"\n"
            "  ros2 service call /ur_dual/execute_last_plan std_srvs/srv/SetBool \"{data: false}\""
            "  ros2 service call /ur_dual/plan_pregrasp_cartesian_from_latest_pose std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/plan_ready_right std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/clear_octomap std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/freeze_octomap std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/unfreeze_octomap std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/attach_grasped_object std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/detach_grasped_object std_srvs/srv/Trigger \"{}\"\n"
            "  ros2 service call /ur_dual/clear_octomap_around_object std_srvs/srv/Trigger \"{}\"\n"
        )

    def _on_joint_state(self, msg: JointState):
        self.latest_joint_state = msg


    def _on_object_pose(self, msg: PoseStamped):
        """Guarda la última pose de objeto recibida.

        La pose debe llegar ya transformada al frame base del robot.
        El log se limita en frecuencia para no saturar la consola mientras
        el bridge publica continuamente detecciones.
        """

        if not msg.header.frame_id:
            self.get_logger().warn("Pose de objeto recibida sin frame_id. Se ignora.")
            return

        self.latest_object_pose = msg

        now = time.monotonic()
        if now - self.last_object_pose_log_time < 1.5:
            return

        self.last_object_pose_log_time = now
        self.get_logger().info(
            "Pose de objeto recibida:\n"
            f"  frame_id: {msg.header.frame_id}\n"
            f"  x={msg.pose.position.x:.3f}, "
            f"y={msg.pose.position.y:.3f}, "
            f"z={msg.pose.position.z:.3f}"
        )


    def _wait_for_joint_state(self, timeout_s=5.0) -> bool:
        start = time.monotonic()

        while time.monotonic() - start < timeout_s:
            if self.latest_joint_state is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    def _publish_plan_for_rviz(self, plan_result, n_times=5, period_s=0.5):
        if self.latest_joint_state is None:
            self.get_logger().warn(
                "No hay /joint_states todavía. Publicaré trayectoria sin trajectory_start."
            )
            trajectory_start = RobotState()
        else:
            trajectory_start = RobotState()
            trajectory_start.joint_state = self.latest_joint_state
            trajectory_start.is_diff = False

        traj_msg = plan_result.trajectory.get_robot_trajectory_msg()

        display_msg = DisplayTrajectory()
        display_msg.model_id = "ur_dual"
        display_msg.trajectory_start = trajectory_start
        display_msg.trajectory.append(traj_msg)

        for i in range(n_times):
            self.display_pub.publish(display_msg)
            self.get_logger().info(
                f"Plan publicado en /display_planned_path ({i + 1}/{n_times})."
            )
            time.sleep(period_s)


    def _make_pregrasp_pose_from_latest_object(self) -> PoseStamped | None:
        """Crea una pose pre-grasp desde la última pose de objeto.

        Para objetos normales:
        - El TCP lógico es qbhand2m1_palm_link.
        - La posición se calcula como object_pose + offset aprendido.
        - La orientación se toma de la pose enseñada manualmente.
        """

        if self.latest_object_pose is None:
            self.get_logger().error(
                "No hay pose de objeto todavía. Espera a que /ur_dual/object_pose publique."
            )
            return None

        object_pose = self.latest_object_pose

        base_frame = self.get_parameter("pregrasp_base_frame").value
        pose_link = self.get_parameter("pregrasp_pose_link").value
        orientation_mode = self.get_parameter("pregrasp_orientation_mode").value

        if object_pose.header.frame_id != base_frame:
            self.get_logger().error(
                f"La pose de objeto debe venir en '{base_frame}', "
                f"pero llegó en '{object_pose.header.frame_id}'."
            )
            return None

        pregrasp = PoseStamped()
        pregrasp.header.frame_id = base_frame
        pregrasp.header.stamp = self.get_clock().now().to_msg()

        # ------------------------------------------------------------
        # Modo principal: plantilla aprendida para objetos normales.
        # ------------------------------------------------------------
        if orientation_mode in ("normal_grasp", "grasp_template"):
            grasp_mode = self.get_parameter("grasp_mode").value

            if grasp_mode == "bottle":
                prefix = "bottle_grasp"
                position_mode = "BOTTLE_GRASP_TEMPLATE"
            else:
                prefix = "normal_grasp"
                position_mode = "NORMAL_GRASP_TEMPLATE"

            dx = float(self.get_parameter(f"{prefix}_dx").value)
            dy = float(self.get_parameter(f"{prefix}_dy").value)
            dz = float(self.get_parameter(f"{prefix}_dz").value)

            pregrasp.pose.position.x = object_pose.pose.position.x + dx
            pregrasp.pose.position.y = object_pose.pose.position.y + dy
            pregrasp.pose.position.z = object_pose.pose.position.z + dz

            pregrasp.pose.orientation.x = float(
                self.get_parameter(f"{prefix}_qx").value
            )
            pregrasp.pose.orientation.y = float(
                self.get_parameter(f"{prefix}_qy").value
            )
            pregrasp.pose.orientation.z = float(
                self.get_parameter(f"{prefix}_qz").value
            )
            pregrasp.pose.orientation.w = float(
                self.get_parameter(f"{prefix}_qw").value
            )

            position_mode = (
                f"{position_mode}: object + "
                f"dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}"
            )

        # ------------------------------------------------------------
        # Fallback: usar orientación actual del link.
        # ------------------------------------------------------------
        elif orientation_mode == "current_tf":
            approach_height = float(self.get_parameter("approach_height").value)

            pregrasp.pose.position.x = object_pose.pose.position.x
            pregrasp.pose.position.y = object_pose.pose.position.y
            pregrasp.pose.position.z = object_pose.pose.position.z + approach_height

            try:
                transform = self.tf_buffer.lookup_transform(
                    base_frame,
                    pose_link,
                    Time(),
                    timeout=Duration(seconds=2.0),
                )
                pregrasp.pose.orientation = transform.transform.rotation
            except Exception as exc:
                self.get_logger().error(
                    f"No se pudo leer TF {base_frame} <- {pose_link}: {exc}"
                )
                return None

            position_mode = "CURRENT_TF: object + approach_height"

        # ------------------------------------------------------------
        # Fallback: orientación por RPY.
        # ------------------------------------------------------------
        else:
            palm_z_offset = float(self.get_parameter("palm_z_offset").value)
            approach_height = float(self.get_parameter("approach_height").value)

            pregrasp.pose.position.x = object_pose.pose.position.x
            pregrasp.pose.position.y = object_pose.pose.position.y

            if pose_link == "qbhand2m1_palm_link":
                pregrasp.pose.position.z = object_pose.pose.position.z + approach_height
                position_mode = "PALM_LINK_RPY: object.z + approach_height"
            else:
                pregrasp.pose.position.z = (
                    object_pose.pose.position.z + palm_z_offset + approach_height
                )
                position_mode = "TOOL0_RPY: object.z + palm_z_offset + approach_height"

            roll = float(self.get_parameter("pregrasp_roll").value)
            pitch = float(self.get_parameter("pregrasp_pitch").value)
            yaw = float(self.get_parameter("pregrasp_yaw").value)

            pregrasp.pose.orientation = quaternion_from_rpy(roll, pitch, yaw)

        q = pregrasp.pose.orientation

        self.get_logger().info(
            "Pose pre-grasp generada:\n"
            f"  frame_id: {pregrasp.header.frame_id}\n"
            f"  pose_link: {pose_link}\n"
            f"  x={pregrasp.pose.position.x:.3f}, "
            f"y={pregrasp.pose.position.y:.3f}, "
            f"z={pregrasp.pose.position.z:.3f}\n"
            f"  objeto x={object_pose.pose.position.x:.3f}, "
            f"y={object_pose.pose.position.y:.3f}, "
            f"z={object_pose.pose.position.z:.3f}\n"
            f"  position_mode={position_mode}\n"
            f"  orientation_mode={orientation_mode}\n"
            f"  quat=[x={q.x:.3f}, y={q.y:.3f}, z={q.z:.3f}, w={q.w:.3f}]"
        )

        return pregrasp


    def _make_offset_pose_from_current_tool(self):
        """Lee la pose actual del pose_link por TF y crea una pose objetivo
        desplazada por un offset pequeño.

        Esta prueba valida la planificación hacia PoseStamped sin depender
        todavía del stereo/YOLO.
        """

        base_frame = self.get_parameter("offset_test_base_frame").value
        pose_link = self.get_parameter("offset_test_pose_link").value

        dx = float(self.get_parameter("offset_test_dx").value)
        dy = float(self.get_parameter("offset_test_dy").value)
        dz = float(self.get_parameter("offset_test_dz").value)

        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                pose_link,
                Time(),
                timeout=Duration(seconds=2.0),
            )
        except Exception as exc:
            self.get_logger().error(
                f"No se pudo leer TF {base_frame} <- {pose_link}: {exc}"
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = base_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = transform.transform.translation.x + dx
        pose.pose.position.y = transform.transform.translation.y + dy
        pose.pose.position.z = transform.transform.translation.z + dz

        # Conservamos la misma orientación actual del tool0.
        # Así evitamos IK raro o giros inesperados en esta primera prueba.
        pose.pose.orientation = transform.transform.rotation

        self.get_logger().info(
            "Pose offset generada:\n"
            f"  frame_id: {pose.header.frame_id}\n"
            f"  pose_link: {pose_link}\n"
            f"  posición objetivo: "
            f"x={pose.pose.position.x:.3f}, "
            f"y={pose.pose.position.y:.3f}, "
            f"z={pose.pose.position.z:.3f}\n"
            f"  offset aplicado: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}"
        )

        return pose


    def _make_cartesian_pregrasp_waypoints_from_latest_object(self):
        """Genera waypoints cartesianos desde la pose actual hacia el pre-grasp.

        Estrategia:
        1. Mantener orientación actual del tool0.
        2. Mover X/Y sobre el objeto a una altura segura.
        3. Bajar a pre-grasp.
        """

        if self.latest_object_pose is None:
            self.get_logger().error(
                "No hay pose de objeto todavía. Publica primero en /ur_dual/object_pose."
            )
            return None

        object_pose = self.latest_object_pose

        base_frame = "ur_dual_I_base_link"

        if object_pose.header.frame_id != base_frame:
            self.get_logger().error(
                f"Por ahora la pose de objeto debe venir en frame '{base_frame}', "
                f"pero llegó en '{object_pose.header.frame_id}'."
            )
            return None

        pose_link = self.get_parameter("cartesian_pregrasp_pose_link").value

        palm_z_offset = float(self.get_parameter("palm_z_offset").value)
        approach_height = float(self.get_parameter("approach_height").value)
        cartesian_safe_z = float(self.get_parameter("cartesian_safe_z").value)

        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                pose_link,
                Time(),
                timeout=Duration(seconds=2.0),
            )
        except Exception as exc:
            self.get_logger().error(
                f"No se pudo leer TF {base_frame} <- {pose_link}: {exc}"
            )
            return None

        current_x = transform.transform.translation.x
        current_y = transform.transform.translation.y
        current_z = transform.transform.translation.z
        current_orientation = transform.transform.rotation

        final_z = object_pose.pose.position.z + palm_z_offset + approach_height

        # Altura de tránsito:
        # - Si el robot ya está más bajo que cartesian_safe_z, NO lo obligamos a subir.
        # - Si está muy bajo, lo mantenemos al menos 10 cm sobre el pre-grasp.
        min_transit_z = final_z + 0.10
        transit_z = min(current_z, cartesian_safe_z)
        transit_z = max(transit_z, min_transit_z)

        self.get_logger().info(
            f"Altura de tránsito calculada: current_z={current_z:.3f}, "
            f"cartesian_safe_z={cartesian_safe_z:.3f}, "
            f"final_z={final_z:.3f}, transit_z={transit_z:.3f}"
        )


        waypoints = []

        # Waypoint 1: bajar en Z desde la pose actual, manteniendo X/Y actuales.
        waypoint_down = PoseStamped()
        waypoint_down.header.frame_id = base_frame
        waypoint_down.header.stamp = self.get_clock().now().to_msg()
        waypoint_down.pose.position.x = current_x
        waypoint_down.pose.position.y = current_y
        waypoint_down.pose.position.z = transit_z
        waypoint_down.pose.orientation = current_orientation
        waypoints.append(waypoint_down)

        # Waypoint 2: moverse en X/Y hacia encima del objeto, manteniendo altura de tránsito.
        waypoint_xy = PoseStamped()
        waypoint_xy.header.frame_id = base_frame
        waypoint_xy.header.stamp = self.get_clock().now().to_msg()
        waypoint_xy.pose.position.x = object_pose.pose.position.x
        waypoint_xy.pose.position.y = object_pose.pose.position.y
        waypoint_xy.pose.position.z = transit_z
        waypoint_xy.pose.orientation = current_orientation
        waypoints.append(waypoint_xy)

        # Waypoint 3: bajar a pre-grasp.
        waypoint_pregrasp = PoseStamped()
        waypoint_pregrasp.header.frame_id = base_frame
        waypoint_pregrasp.header.stamp = self.get_clock().now().to_msg()
        waypoint_pregrasp.pose.position.x = object_pose.pose.position.x
        waypoint_pregrasp.pose.position.y = object_pose.pose.position.y
        waypoint_pregrasp.pose.position.z = final_z
        waypoint_pregrasp.pose.orientation = current_orientation
        waypoints.append(waypoint_pregrasp)

        self.get_logger().info(
            "Waypoints cartesianos pre-grasp generados:\n"
            f"  pose_link: {pose_link}\n"
            f"  current xyz: x={current_x:.3f}, y={current_y:.3f}, z={current_z:.3f}\n"
            f"  objeto xyz: x={object_pose.pose.position.x:.3f}, "
            f"y={object_pose.pose.position.y:.3f}, z={object_pose.pose.position.z:.3f}\n"
            f"  waypoint down: x={waypoint_down.pose.position.x:.3f}, "
            f"y={waypoint_down.pose.position.y:.3f}, "
            f"z={waypoint_down.pose.position.z:.3f}\n"
            f"  waypoint XY: x={waypoint_xy.pose.position.x:.3f}, "
            f"y={waypoint_xy.pose.position.y:.3f}, "
            f"z={waypoint_xy.pose.position.z:.3f}\n"
            f"  waypoint pregrasp: x={waypoint_pregrasp.pose.position.x:.3f}, "
            f"y={waypoint_pregrasp.pose.position.y:.3f}, "
            f"z={waypoint_pregrasp.pose.position.z:.3f}\n"
            f"  palm_z_offset={palm_z_offset:.3f}, "
            f"approach_height={approach_height:.3f}, "
            f"cartesian_safe_z/transit_z={transit_z:.3f}"
        )

        return waypoints, pose_link


    def _on_plan_pregrasp_cartesian_from_latest_pose(self, request, response):
        """Planifica hacia pre-grasp usando solo Cartesian Path por waypoints."""

        self.get_logger().info(
            "Servicio recibido: plan_pregrasp_cartesian_from_latest_pose"
        )

        self.pending_plan_result = None

        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = (
                "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            )
            return response

        result = self._make_cartesian_pregrasp_waypoints_from_latest_object()

        if result is None:
            response.success = False
            response.message = "No se pudieron generar waypoints cartesianos."
            return response

        waypoints, pose_link = result

        success, status, plan_result = self.cartesian_plan_through_poses(
            arm="right",
            poses=waypoints,
            pose_link=pose_link,
            max_step=0.005,
            jump_threshold=0.0,
            min_fraction=0.95,
            timeout_s=15.0,
            check_cable=True,
            start_joint_state_msg=self.latest_joint_state,
        )

        if not success:
            self.pending_plan_result = None

            if status == "CABLE_MOTION_TOO_LARGE" and plan_result is not None:
                self.get_logger().warn(
                    "Plan cartesiano rechazado por cable. "
                    "Se publicará en RViz solo para diagnóstico. NO queda ejecutable."
                )
                self._publish_plan_for_rviz(plan_result, n_times=5, period_s=0.5)

                response.success = False
                response.message = (
                    "Plan cartesiano encontrado pero rechazado por cable. "
                    "Publicado en RViz solo para diagnóstico. NO ejecutar."
                )
                return response

            response.success = False
            response.message = f"Plan cartesiano pre-grasp falló. Status: {status}"
            return response

        self.pending_plan_result = plan_result
        self._publish_plan_for_rviz(plan_result, n_times=5, period_s=0.5)

        response.success = True
        response.message = (
            "Plan cartesiano pre-grasp generado y publicado en RViz. "
            "Revisa trayectoria antes de ejecutar."
        )
        return response


    def _on_plan_pregrasp_from_latest_pose(self, request, response):
        """Planifica desde el estado actual hacia una pose pre-grasp.

        Este movimiento usa OMPL, pero queda protegido por el chequeo general
        de joints/cable que está en ur_dual_moveit_py.py.
        """

        self.get_logger().info("Servicio recibido: plan_pregrasp_from_latest_pose")
        
        # Descartamos planes anteriores para evitar ejecutar trayectorias viejas.
        self.pending_plan_result = None

        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = (
                "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            )
            return response

        pregrasp_pose = self._make_pregrasp_pose_from_latest_object()

        if pregrasp_pose is None:
            response.success = False
            response.message = "No se pudo generar la pose pre-grasp."
            return response

        pose_link = self.get_parameter("pregrasp_pose_link").value




        # Tolerancias del OrientationConstraint según clase YOLO.
        # Si no hay clase, default a roll/pitch estrictos + yaw libre.
        obj_class = self._latest_object_class
        if obj_class in GRASP_TOLERANCES:
            tol_roll, tol_pitch, tol_yaw = GRASP_TOLERANCES[obj_class]
            tol_src = f"clase '{obj_class}'"
        else:
            tol_roll, tol_pitch, tol_yaw = GRASP_TOLERANCE_DEFAULT
            tol_src = f"default (clase '{obj_class}' no en GRASP_TOLERANCES)"

        self.get_logger().info(
            f"Pregrasp con OrientationConstraint [{tol_src}]: "
            f"tol=({tol_roll:.2f},{tol_pitch:.2f},{tol_yaw:.2f}) rad"
        )



        success, status, plan_result = self.arm_go_to_pose_best_of_n(
            arm="right",
            pose=pregrasp_pose,
            pose_link=pose_link,
            attempts=12,
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
            orientation_tolerance=(tol_roll, tol_pitch, tol_yaw),
            position_tolerance=0.005,
        )

        if not success:
            self.pending_plan_result = None

            # Caso importante:
            # MoveIt sí encontró una trayectoria, pero nuestro filtro de cable
            # la rechazó. La publicamos SOLO para verla en RViz, pero NO queda
            # ejecutable.
            if status == "CABLE_MOTION_TOO_LARGE" and plan_result is not None:
                self.get_logger().warn(
                    "El plan fue rechazado por protección de cable, "
                    "pero se publicará en RViz solo para diagnóstico. "
                    "NO queda pendiente para ejecución."
                )

                self._publish_plan_for_rviz(plan_result, n_times=5, period_s=0.5)

                response.success = False
                response.message = (
                    "Plan encontrado pero rechazado por cable. "
                    "Publicado en RViz solo para diagnóstico. NO ejecutar."
                )
                return response

            response.success = False
            response.message = f"Plan pre-grasp falló. Status: {status}"
            return response

        if plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = "Plan pre-grasp falló: plan_result vacío."
            return response

        self.pending_plan_result = plan_result

        # Menos repeticiones para evitar que el servicio tarde demasiado en responder.
        self._publish_plan_for_rviz(plan_result, n_times=3, period_s=0.2)

        response.success = True
        response.message = (
            "Plan pre-grasp generado y publicado en RViz. "
            "Revisa que los joints no giren demasiado antes de ejecutar."
        )
        return response


    def _on_plan_offset_test(self, request, response):
        """Servicio de prueba hacia una pose cartesiana cercana.

        Planifica desde la pose actual hacia una pose generada por TF + offset.
        No ejecuta automáticamente.
        """

        self.get_logger().info("Servicio recibido: plan_offset_test")
        # Al iniciar cualquier nuevo plan, descartamos planes anteriores.
        self.pending_plan_result = None

        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = (
                "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            )
            return response

        pose = self._make_offset_pose_from_current_tool()

        if pose is None:
            response.success = False
            response.message = "No se pudo generar la pose offset por TF."
            return response

        pose_link = self.get_parameter("offset_test_pose_link").value

        success, status, plan_result = self.arm_go_to_pose_cartesian(
            arm="right",
            pose=pose,
            pose_link=pose_link,
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        if not success or plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = f"Plan cartesiano offset falló. Status: {status}"
            return response

        self.pending_plan_result = plan_result
        self._publish_plan_for_rviz(plan_result, n_times=5, period_s=0.5)

        response.success = True
        response.message = (
            "Plan offset generado y publicado en RViz. "
            "Revisa la trayectoria antes de ejecutar."
        )
        return response


    def _on_plan_home_right(self, request, response):
        self.get_logger().info("Servicio recibido: plan_home_right")
        # Al iniciar cualquier nuevo plan, descartamos planes anteriores.
        # Así evitamos tener que llamar execute_last_plan con data=false manualmente.
        self.pending_plan_result = None
        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            return response

        success, status, plan_result = self.arm_go_to_named_pose(
            arm="right",
            pose_name="Home_Right",
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        if not success or plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = f"Plan falló. Status: {status}"
            return response

        self.pending_plan_result = plan_result
        self._publish_plan_for_rviz(plan_result, n_times=5, period_s=0.5)

        response.success = True
        response.message = (
            "Plan hacia Home_Right generado y publicado en RViz. "
            "Revisa la trayectoria antes de ejecutar."
        )
        return response

    def _on_plan_ready_right(self, request, response):
        """Planifica hacia la pose Ready_Right del brazo derecho."""

        self.get_logger().info("Servicio recibido: plan_ready_right")

        self.pending_plan_result = None

        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = (
                "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            )
            return response

        success, status, plan_result = self.arm_go_to_named_pose(
            arm="right",
            pose_name="Ready_Right",
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        if not success or plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = f"Plan Ready_Right falló. Status: {status}"
            return response

        self.pending_plan_result = plan_result
        self._publish_plan_for_rviz(plan_result, n_times=3, period_s=0.2)

        response.success = True
        response.message = (
            "Plan Ready_Right generado y publicado en RViz. "
            "Revisa antes de ejecutar."
        )
        return response



    def _publish_hand_command(
        self,
        synergy: float,
        manipulation: float,
        duration_s: int = 2,
    ):
        msg = JointTrajectory()
        msg.joint_names = [
            "qbhand2m1_synergy_joint",
            "qbhand2m1_manipulation_joint",
        ]

        point = JointTrajectoryPoint()
        point.positions = [synergy, manipulation]
        point.time_from_start.sec = duration_s

        msg.points.append(point)
        self.hand_pub.publish(msg)

    def _on_open_hand(self, request, response):
        self.get_logger().info("Servicio recibido: open_hand")
        self._publish_hand_command(synergy=0.0, manipulation=0.0, duration_s=2)
        response.success = True
        response.message = "Comando de abrir mano publicado."
        return response

    def _on_close_hand(self, request, response):
        self.get_logger().info("Servicio recibido: close_hand")
        self._publish_hand_command(synergy=0.85, manipulation=0.0, duration_s=2)
        response.success = True
        response.message = "Comando de cerrar mano publicado."
        return response

    def _plan_named_pose_service(self, pose_name: str, response):
        self.pending_plan_result = None

        if not self._wait_for_joint_state(timeout_s=5.0):
            response.success = False
            response.message = (
                "No se recibió /joint_states. ¿Está corriendo start_robot.launch.py?"
            )
            return response

        success, status, plan_result = self.arm_go_to_named_pose(
            arm="right",
            pose_name=pose_name,
            execute=False,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        if not success or plan_result is None:
            self.pending_plan_result = None
            response.success = False
            response.message = f"Plan {pose_name} falló. Status: {status}"
            return response

        self.pending_plan_result = plan_result
        self._publish_plan_for_rviz(plan_result, n_times=3, period_s=0.2)

        response.success = True
        response.message = f"Plan {pose_name} generado y publicado en RViz."
        return response

    def _on_plan_place_normal(self, request, response):
        self.get_logger().info("Servicio recibido: plan_place_normal")
        return self._plan_named_pose_service("Place_Normal_Right", response)

    def _on_plan_place_bottle(self, request, response):
        self.get_logger().info("Servicio recibido: plan_place_bottle")
        return self._plan_named_pose_service("Place_Bottle_Right", response)





    def _update_obstacles_before_planning(self, timeout_s=5.0):
            """
            Llama al servicio obstacle_clusterer para refrescar los obstáculos
            en el planning scene ANTES de cada planificación. Esto da una captura
            'frozen' del mundo: las cajas de colisión generadas por percepción 3D
            quedan fijas durante la planificación, sin race conditions.
            """

            if not self.update_obstacles_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn(
                    "/ur_dual/update_obstacles no disponible; planifico SIN actualizar obstáculos."
                )
                return False

            req = Trigger.Request()
            future = self.update_obstacles_client.call_async(req)

            start = time.monotonic()
            while rclpy.ok() and not future.done():
                if time.monotonic() - start > timeout_s:
                    self.get_logger().warn("Timeout esperando /ur_dual/update_obstacles.")
                    return False
                time.sleep(0.05)

            result = future.result()
            if result is None:
                self.get_logger().warn("update_obstacles devolvió None.")
                return False

            self.get_logger().info(f"update_obstacles: {result.message}")
            # Delay corto para que el planning scene monitor procese el diff
            # antes de que MoveIt empiece a planificar.
            time.sleep(0.3)
            return result.success







    def _call_oak_service(self, client, service_label, timeout_s=3.0):
        """Helper para llamar a un servicio Trigger de la OAK con timeout."""
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                f"Servicio {service_label} no disponible (cámara no corriendo?)."
            )
            return False

        req = Trigger.Request()
        future = client.call_async(req)

        start = time.monotonic()
        while rclpy.ok() and not future.done():
            if time.monotonic() - start > timeout_s:
                self.get_logger().error(f"Timeout llamando a {service_label}.")
                return False
            time.sleep(0.05)

        result = future.result()
        if result is None or not result.success:
            self.get_logger().error(
                f"{service_label} respondió fallo: "
                f"{result.message if result else 'sin respuesta'}"
            )
            return False

        return True

    def _on_freeze_octomap(self, request, response):
        """
        Congela el OctoMap deteniendo el pipeline de la OAK.
        El OctoMap actual queda como snapshot durante la ejecución del
        pick-and-place, evitando que el brazo, la mano o el cable de la
        SoftHand sean incorporados como obstáculos espurios.
        """
        if self._octomap_frozen:
            response.success = True
            response.message = "OctoMap ya estaba congelado."
            return response

        self.get_logger().info(
            "FREEZE OctoMap: deteniendo pipeline de la OAK..."
        )

        ok = self._call_oak_service(self.oak_stop_client, "/oak_cam/oak/stop_camera")
        if not ok:
            response.success = False
            response.message = "No pude detener la cámara para congelar OctoMap."
            return response

        # Pequeño delay para que el plugin DepthImageOctomapUpdater procese
        # los últimos frames que ya tenía en cola antes de quedar sin datos.
        time.sleep(0.5)

        self._octomap_frozen = True
        response.success = True
        response.message = "OctoMap congelado (cámara detenida)."
        self.get_logger().info(response.message)
        return response

    def _on_unfreeze_octomap(self, request, response):
        """
        Reanuda la captura del OctoMap rearrancando el pipeline de la OAK.
        Se llama al final de la secuencia de pick-and-place, ya con el
        brazo retornado a Ready_Right.
        """
        if not self._octomap_frozen:
            response.success = True
            response.message = "OctoMap ya estaba activo."
            return response

        self.get_logger().info(
            "UNFREEZE OctoMap: rearrancando pipeline de la OAK..."
        )

        ok = self._call_oak_service(self.oak_start_client, "/oak_cam/oak/start_camera")
        if not ok:
            response.success = False
            response.message = "No pude rearrancar la cámara para descongelar OctoMap."
            return response

        # Tras start_camera la OAK tarda ~1-2s en estabilizar.
        # No bloqueamos aquí; el siguiente ciclo continuará normalmente.
        self._octomap_frozen = False
        response.success = True
        response.message = "OctoMap descongelado (cámara reiniciada)."
        self.get_logger().info(response.message)
        return response






    def _get_softhand_touch_links(self):
        """
        Lista de links de la SoftHand que MoveIt debe permitir tocar
        al attached object sin marcarlo como colisión.

        Estos nombres vienen del URDF del qbhand2m1. Si en el futuro
        cambia la SoftHand, hay que revisar y actualizar estos nombres.
        """
        return [
            "qbhand2m1_palm_link",
            "qbhand2m1_thumb_knuckle_link",
            "qbhand2m1_thumb_proximal_link",
            "qbhand2m1_thumb_distal_link",
            "qbhand2m1_index_knuckle_link",
            "qbhand2m1_index_proximal_link",
            "qbhand2m1_index_middle_link",
            "qbhand2m1_index_distal_link",
            "qbhand2m1_middle_knuckle_link",
            "qbhand2m1_middle_proximal_link",
            "qbhand2m1_middle_middle_link",
            "qbhand2m1_middle_distal_link",
            "qbhand2m1_ring_knuckle_link",
            "qbhand2m1_ring_proximal_link",
            "qbhand2m1_ring_middle_link",
            "qbhand2m1_ring_distal_link",
            "qbhand2m1_little_knuckle_link",
            "qbhand2m1_little_proximal_link",
            "qbhand2m1_little_middle_link",
            "qbhand2m1_little_distal_link",
        ]

    def _on_attach_grasped_object(self, request, response):
        """
        Agrega el objeto agarrado como AttachedCollisionObject al frame
        de la mano. A partir de este punto, MoveIt considera la geometría
        conjunta (mano + objeto) para todas las planificaciones, evitando
        colisiones del objeto contra obstáculos del entorno durante
        lift y place.
        """
        if self._object_attached:
            response.success = True
            response.message = "El objeto ya estaba attached."
            return response

        obj_id = self.get_parameter("attached_object_id").value
        frame = self.get_parameter("attached_object_frame").value
        sx = float(self.get_parameter("attached_object_size_x").value)
        sy = float(self.get_parameter("attached_object_size_y").value)
        sz = float(self.get_parameter("attached_object_size_z").value)
        ox = float(self.get_parameter("attached_object_offset_x").value)
        oy = float(self.get_parameter("attached_object_offset_y").value)
        oz = float(self.get_parameter("attached_object_offset_z").value)

        self.get_logger().info(
            f"ATTACH '{obj_id}' a '{frame}': "
            f"size=({sx:.3f},{sy:.3f},{sz:.3f}), offset=({ox:.3f},{oy:.3f},{oz:.3f})"
        )

        # Construir CollisionObject base.
        collision = CollisionObject()
        collision.header.frame_id = frame
        collision.header.stamp = self.get_clock().now().to_msg()
        collision.id = obj_id
        collision.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [sx, sy, sz]

        pose = Pose()
        pose.position.x = ox
        pose.position.y = oy
        pose.position.z = oz
        pose.orientation.w = 1.0

        collision.primitives.append(box)
        collision.primitive_poses.append(pose)

        # Construir AttachedCollisionObject.
        attached = AttachedCollisionObject()
        attached.link_name = frame
        attached.object = collision
        attached.touch_links = self._get_softhand_touch_links()

        # Publicar en planning_scene como diff.
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)

        self.planning_scene_pub.publish(scene)

        # Pequeño delay para que el PlanningSceneMonitor procese el diff.
        time.sleep(0.3)

        self._object_attached = True
        response.success = True
        response.message = f"Objeto '{obj_id}' attached a '{frame}'."
        self.get_logger().info(response.message)
        return response

    def _on_detach_grasped_object(self, request, response):
        """
        Quita el AttachedCollisionObject del frame de la mano.
        Llamar JUSTO ANTES de abrir la mano en el place.
        Después de detach, MoveIt no considera el objeto como parte del robot.
        """
        if not self._object_attached:
            response.success = True
            response.message = "No hay objeto attached para liberar."
            return response

        obj_id = self.get_parameter("attached_object_id").value
        frame = self.get_parameter("attached_object_frame").value

        self.get_logger().info(f"DETACH '{obj_id}' de '{frame}'.")

        # CollisionObject con operation REMOVE.
        collision = CollisionObject()
        collision.id = obj_id
        collision.operation = CollisionObject.REMOVE
        collision.header.frame_id = frame

        attached = AttachedCollisionObject()
        attached.link_name = frame
        attached.object = collision

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)

        self.planning_scene_pub.publish(scene)

        # También removerlo del mundo en caso de que quede ahí flotando.
        time.sleep(0.2)
        world_remove = CollisionObject()
        world_remove.id = obj_id
        world_remove.operation = CollisionObject.REMOVE
        world_remove.header.frame_id = frame

        scene2 = PlanningScene()
        scene2.is_diff = True
        scene2.world.collision_objects.append(world_remove)
        self.planning_scene_pub.publish(scene2)

        time.sleep(0.3)

        self._object_attached = False
        response.success = True
        response.message = f"Objeto '{obj_id}' detached."
        self.get_logger().info(response.message)
        return response




    def _on_object_class(self, msg):
        self._latest_object_class = msg.data

    def _on_clear_octomap_around_object(self, request, response):
        """
        Limpia el OctoMap dentro de una caja del tamaño aproximado del
        objeto detectado, centrada en su pose. La caja se elige según
        la clase YOLO del último objeto recibido (OBJECT_DIMENSIONS).

        MoveIt procesa el CollisionObject REMOVE con geometría y borra
        los voxels del OctoMap que caen dentro. El warning
        'Tried to remove world object 'octomap_clear_region', but it
        does not exist in this scene' es inofensivo: lo emite MoveIt
        antes de procesar el efecto sobre el OctoMap, pero la limpieza
        de voxels sí ocurre.

        Llamar este servicio DESPUÉS de freeze_octomap para que los
        voxels limpios no se repueblen mientras se planifica.
        """
        if self.latest_object_pose is None:
            response.success = False
            response.message = "No hay pose de objeto disponible."
            return response

        margin = float(self.get_parameter("clear_box_margin").value)

        obj_class = self._latest_object_class
        if obj_class in OBJECT_DIMENSIONS:
            dx, dy, dz = OBJECT_DIMENSIONS[obj_class]
            src = f"clase '{obj_class}'"
        else:
            dx, dy, dz = OBJECT_DIM_DEFAULT
            src = f"default (clase '{obj_class}' no en diccionario)"

        sx = dx + 2.0 * margin
        sy = dy + 2.0 * margin
        sz = dz + 2.0 * margin

        pose = self.latest_object_pose

        self.get_logger().info(
            f"CLEAR OctoMap caja para {src}: "
            f"size=({sx:.3f},{sy:.3f},{sz:.3f}) m centrada en "
            f"({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},"
            f"{pose.pose.position.z:.3f})"
        )

        # CollisionObject tipo BOX con operation REMOVE. MoveIt limpia
        # los voxels del octomap que caen dentro de esta caja.
        collision = CollisionObject()
        collision.header.frame_id = pose.header.frame_id
        collision.header.stamp = self.get_clock().now().to_msg()
        collision.id = "octomap_clear_region"
        collision.operation = CollisionObject.REMOVE

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [sx, sy, sz]

        box_pose = Pose()
        box_pose.position = pose.pose.position
        box_pose.orientation.w = 1.0

        collision.primitives.append(box)
        collision.primitive_poses.append(box_pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(collision)

        self.planning_scene_pub.publish(scene)
        time.sleep(0.3)

        response.success = True
        response.message = (
            f"Limpieza solicitada ({src}): caja "
            f"{sx:.3f}x{sy:.3f}x{sz:.3f}m."
        )
        return response








    def _on_clear_octomap(self, request, response):
        """Limpia el OctoMap local del commander."""

        self.get_logger().info("Servicio recibido: clear_octomap local del commander")

        success, message = self.clear_local_octomap()

        response.success = success
        response.message = message
        return response





    def _on_execute_last_plan(self, request, response):
        if request.data is False:
            self.pending_plan_result = None
            response.success = True
            response.message = "Plan pendiente descartado."
            return response

        if self.pending_plan_result is None:
            response.success = False
            response.message = "No hay plan pendiente. Primero llama un servicio de planificación."
            return response

        plan_to_execute = self.pending_plan_result

        # Apenas se solicita ejecución, borramos el plan pendiente.
        # Así no se reutiliza por accidente una trayectoria vieja.
        self.pending_plan_result = None

        self.get_logger().warn(
            "Ejecutando último plan. Vigila el área de trabajo y el paro de emergencia."
        )

        result = self.execute_trajectory(
            plan_to_execute.trajectory,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
        )

        status_text = str(getattr(result, "status", result))

        response.success = "SUCCEEDED" in status_text or "SUCCESS" in status_text or bool(result)
        response.message = f"Resultado ejecución: {status_text}"

        return response


def main():
    rclpy.init()

    node = UrDualCommandServices(name="ur_dual_commander")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()