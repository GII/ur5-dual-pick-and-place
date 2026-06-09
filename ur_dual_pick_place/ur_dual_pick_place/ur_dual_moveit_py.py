#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# ur_dual_moveit_py.py
#
# Wrapper principal de MoveItPy para el sistema UR5 dual + SoftHand.
#
# Inspirado en la arquitectura de tiago_dual_moveit_py, pero reducido a lo que
# necesitamos ahora:
#   - Solo brazo derecho lógico: grupo MoveIt "Right_arm"
#   - Planificación a named poses
#   - Planificación a PoseStamped
#   - Ejecución opcional
#   - Métodos base para collision objects
#
# IMPORTANTE:
# El nombre del nodo ROS2 y el node_name de MoveItPy deben coincidir.
# Si no coinciden, los parámetros de motion_planning.yaml caen en otro namespace
# y MoveItPy no carga bien los planning pipelines.
# -----------------------------------------------------------------------------

import time
import rclpy
import threading
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import GetCartesianPath
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from moveit.planning import MoveItPy, PlanningComponent
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.robot_state import RobotState, robotStateToRobotStateMsg
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState as RobotStateMsg
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
)


class UrDualMoveItPy(Node):
    """Wrapper base para controlar el UR5 dual desde MoveItPy."""

    def __init__(self, name: str = "ur_dual_commander"):
        super().__init__(name)

        self.get_logger().info("Inicializando UrDualMoveItPy...")

        # MoveItPy debe usar el mismo nombre del nodo.
        self.robot = MoveItPy(node_name=name)

        # Modelo y planning scene monitor.
        self.robot_model = self.robot.get_robot_model()
        self.planning_monitor = self.robot.get_planning_scene_monitor()

        # Configuración de ejecución, siguiendo el patrón de TIAGo.
        try:
            trajectory_execution = self.robot.get_trajectory_execution_manager()
            trajectory_execution.enable_execution_duration_monitoring(True)
            trajectory_execution.set_allowed_execution_duration_scaling(1.2)
            trajectory_execution.set_allowed_start_tolerance(0.05)
            self.get_logger().info("TrajectoryExecutionManager configurado.")
        except Exception as exc:
            self.get_logger().warn(
                f"No se pudo configurar TrajectoryExecutionManager: {exc}"
            )

        # Grupos MoveIt. Por ahora usamos solo Right_arm.
        # En tu configuración, Right_arm corresponde físicamente al brazo con prefijo ur_dual_I_*.
        self.groups: Dict[str, PlanningComponent] = {
            "right": self.robot.get_planning_component("Right_arm"),
        }

        # Link que se usará como TCP para goals cartesianos.
        # Para la primera validación dejamos tool0 porque sabemos que está dentro del grupo.
        # Luego probamos qbhand2m1_palm_link si queremos que el target sea la palma.
        self.pose_links: Dict[str, str] = {
            "right": "ur_dual_I_tool0",
            # Alternativa futura:
            # "right": "qbhand2m1_palm_link",
        }

        # Publisher para diffs de planning scene.
        self.planning_scene_pub = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            10,
        )



        self._octomap_synced_once = False
        self._octomap_sync_sub = self.create_subscription(
            PlanningScene,
            "/monitored_planning_scene",
            self._on_move_group_scene,
            10,
        )














        # Callback group dedicado para clientes internos, siguiendo el patrón
        # de tiago_dual_moveit_py. Esto evita bloqueos cuando un servicio
        # nuestro llama a otro servicio de ROS2.
        self.cb_internal_client = MutuallyExclusiveCallbackGroup()

        self.cartesian_plan_client = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
            callback_group=self.cb_internal_client,
        )

        self.get_logger().info("UrDualMoveItPy listo.")

    # -------------------------------------------------------------------------
    # Helpers internos
    # -------------------------------------------------------------------------

    def _get_arm(self, arm: str) -> PlanningComponent:
        if arm not in self.groups:
            raise ValueError(f"Brazo inválido: {arm}. Opciones: {list(self.groups)}")
        return self.groups[arm]

    def _get_pose_link(self, arm: str, pose_link: Optional[str] = None) -> str:
        return pose_link if pose_link is not None else self.pose_links[arm]

    # -------------------------------------------------------------------------
    # Planificación general
    # -------------------------------------------------------------------------

    def plan(self, planning_component: PlanningComponent):
        """Ejecuta planning_component.plan() y devuelve el resultado."""
        self.get_logger().info("Planificando trayectoria...")
        return planning_component.plan()

    def execute_trajectory(
        self,
        trajectory: RobotTrajectory,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
        sleep_time: float = 0.0,
    ):
        """Ejecuta una RobotTrajectory usando los controladores configurados."""

        self.get_logger().warn(
            "Ejecutando trayectoria. Mantén vigilancia y paro de emergencia listo."
        )

        try:
            trajectory.apply_totg_time_parameterization(
                velocity_scaling,
                acceleration_scaling,
            )
        except Exception as exc:
            self.get_logger().warn(
                f"No se pudo aplicar TOTG manualmente. Se ejecutará la trayectoria como viene: {exc}"
            )

        result = self.robot.execute(trajectory, controllers=[])
        time.sleep(sleep_time)

        self.get_logger().info(f"Resultado de ejecución: {result}")
        return result


    def _trajectory_cable_motion_ok(
        self,
        trajectory: RobotTrajectory,
        max_delta_by_joint: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Revisa que ningún joint del brazo derecho gire demasiado.

        Esto no reemplaza las colisiones de MoveIt. Es una protección extra
        por el cable de la SoftHand, porque OMPL puede generar una trayectoria
        válida pero con vueltas innecesarias.
        """

        if max_delta_by_joint is None:
            max_delta_by_joint = {
                "ur_dual_I_shoulder_pan_joint": 3.80,
                "ur_dual_I_shoulder_lift_joint": 3.40,
                "ur_dual_I_elbow_joint": 3.40,

                "ur_dual_I_wrist_1_joint": 1.70,
                "ur_dual_I_wrist_2_joint": 1.55,
                "ur_dual_I_wrist_3_joint": 0.75,
            }

        traj_msg = trajectory.get_robot_trajectory_msg()
        joint_names = list(traj_msg.joint_trajectory.joint_names)

        if not traj_msg.joint_trajectory.points:
            self.get_logger().error("Trayectoria sin puntos. Se rechaza por seguridad.")
            return False

        ok = True

        self.get_logger().info("Chequeo general de cable/joints:")

        for joint_name, max_delta_rad in max_delta_by_joint.items():
            if joint_name not in joint_names:
                self.get_logger().warn(
                    f"  {joint_name}: no aparece en la trayectoria."
                )
                continue

            idx = joint_names.index(joint_name)

            positions = [
                point.positions[idx]
                for point in traj_msg.joint_trajectory.points
                if len(point.positions) > idx
            ]

            if not positions:
                self.get_logger().warn(
                    f"  {joint_name}: sin posiciones en la trayectoria."
                )
                continue

            delta = max(positions) - min(positions)
            delta_deg = delta * 57.2958
            max_delta_deg = max_delta_rad * 57.2958

            self.get_logger().info(
                f"  {joint_name}: delta={delta:.3f} rad "
                f"({delta_deg:.1f} deg), límite={max_delta_rad:.3f} rad "
                f"({max_delta_deg:.1f} deg)"
            )

            if abs(delta) > max_delta_rad:
                ok = False
                self.get_logger().error(
                    f"  RECHAZADO: {joint_name} gira demasiado."
                )

        if not ok:
            self.get_logger().error(
                "Trayectoria rechazada por protección de cable."
            )

        return ok


    def _trajectory_cable_cost(self, trajectory: RobotTrajectory) -> tuple[float, dict]:
        """Calcula un costo de trayectoria según cuánto giran los joints.

        Menor costo = trayectoria más amable con el cable.

        Este costo NO decide colisiones. MoveIt ya valida colisiones antes.
        Solo sirve para escoger entre varias trayectorias válidas.
        """

        traj_msg = trajectory.get_robot_trajectory_msg()
        joint_names = list(traj_msg.joint_trajectory.joint_names)

        # Pesos: muñeca pesa más porque el cable de la SoftHand sufre más ahí.
        weights = {
            "ur_dual_I_shoulder_pan_joint": 0.001,
            "ur_dual_I_shoulder_lift_joint": 0.001,
            "ur_dual_I_elbow_joint": 0.001,
            "ur_dual_I_wrist_1_joint": 0.001,
            "ur_dual_I_wrist_2_joint": 0.001,
            "ur_dual_I_wrist_3_joint": 0.001,
        }

        deltas = {}
        cost = 0.0

        for joint_name, weight in weights.items():
            if joint_name not in joint_names:
                continue

            idx = joint_names.index(joint_name)

            positions = [
                point.positions[idx]
                for point in traj_msg.joint_trajectory.points
                if len(point.positions) > idx
            ]

            if not positions:
                continue

            delta = max(positions) - min(positions)
            deltas[joint_name] = delta
            cost += weight * abs(delta)

        return cost, deltas

    def _log_cable_deltas(self, deltas: dict, prefix: str = ""):
        """Imprime deltas de joints de forma legible."""

        if prefix:
            self.get_logger().info(prefix)

        for joint_name, delta in deltas.items():
            self.get_logger().info(
                f"  {joint_name}: delta={delta:.3f} rad "
                f"({delta * 57.2958:.1f} deg)"
            )


    def arm_plan_to_pose_raw(
        self,
        arm: str,
        pose: PoseStamped,
        pose_link: Optional[str] = None,
        orientation_tolerance: Optional[tuple] = None,
        position_tolerance: float = 0.005,
    ):
        """Planifica hacia una pose y devuelve el plan sin aplicar filtro de cable.

        Se usa para generar candidatos. Luego otro método escoge el mejor.

        Si orientation_tolerance es None, planifica hacia la pose exacta
        con la tolerancia default de MoveIt (comportamiento original).

        Si orientation_tolerance es (tol_roll, tol_pitch, tol_yaw) en
        radianes, construye un goal con:
          - PositionConstraint: esfera de radio position_tolerance
            alrededor del target XYZ.
          - OrientationConstraint: tolerancias por eje en torno a la
            orientación de referencia.
        Esto permite que OMPL explore muchas más soluciones IK,
        especialmente cuando se libera el yaw (eje vertical del agarre).
        """

        selected_arm = self._get_arm(arm)
        link = self._get_pose_link(arm, pose_link)

        selected_arm.set_start_state_to_current_state()

        if orientation_tolerance is None:
            # Comportamiento original: pose exacta.
            selected_arm.set_goal_state(
                pose_stamped_msg=pose,
                pose_link=link,
            )
            mode_msg = "pose exacta"
        else:
            # Goal con constraints (posición tight + orientación tolerante).
            constraints = self._build_pose_constraints(
                pose=pose,
                link=link,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
            selected_arm.set_goal_state(motion_plan_constraints=[constraints])
            tol_r, tol_p, tol_y = orientation_tolerance
            mode_msg = (
                f"constraints (pos={position_tolerance*1000:.0f}mm, "
                f"ori=({tol_r:.2f},{tol_p:.2f},{tol_y:.2f}) rad)"
            )

        self.get_logger().info(
            f"Plan candidato hacia pose usando pose_link='{link}' "
            f"frame='{pose.header.frame_id}' [{mode_msg}]"
        )

        plan_result = selected_arm.plan()

        if not plan_result:
            return None

        return plan_result


    def _build_pose_constraints(
        self,
        pose: PoseStamped,
        link: str,
        position_tolerance: float,
        orientation_tolerance: tuple,
    ) -> Constraints:
        """
        Construye un mensaje Constraints con:
          - PositionConstraint: target XYZ con esfera de tolerancia.
          - OrientationConstraint: orientación de referencia con
            tolerancias por eje (XYZ Euler).

        Esto es lo que OMPL usa como goal cuando le pasamos
        motion_plan_constraints en lugar de pose_stamped_msg.
        """
        constraints = Constraints()
        constraints.name = "pregrasp_target_with_tol"

        # ---- PositionConstraint ----
        pos_c = PositionConstraint()
        pos_c.header = pose.header
        pos_c.link_name = link
        pos_c.target_point_offset.x = 0.0
        pos_c.target_point_offset.y = 0.0
        pos_c.target_point_offset.z = 0.0

        # Región tipo esfera centrada en el target.
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [position_tolerance]
        pos_c.constraint_region.primitives.append(sphere)

        from geometry_msgs.msg import Pose as PoseMsg
        region_pose = PoseMsg()
        region_pose.position = pose.pose.position
        region_pose.orientation.w = 1.0
        pos_c.constraint_region.primitive_poses.append(region_pose)
        pos_c.weight = 1.0

        constraints.position_constraints.append(pos_c)

        # ---- OrientationConstraint ----
        tol_roll, tol_pitch, tol_yaw = orientation_tolerance

        ori_c = OrientationConstraint()
        ori_c.header = pose.header
        ori_c.link_name = link
        ori_c.orientation = pose.pose.orientation
        ori_c.absolute_x_axis_tolerance = float(tol_roll)
        ori_c.absolute_y_axis_tolerance = float(tol_pitch)
        ori_c.absolute_z_axis_tolerance = float(tol_yaw)
        ori_c.parameterization = OrientationConstraint.XYZ_EULER_ANGLES
        ori_c.weight = 1.0

        constraints.orientation_constraints.append(ori_c)

        return constraints


    def arm_go_to_pose_best_of_n(
        self,
        arm: str,
        pose: PoseStamped,
        pose_link: Optional[str] = None,
        attempts: int = 12,
        execute: bool = False,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
        orientation_tolerance: Optional[tuple] = None,
        position_tolerance: float = 0.005,
    ):
        """Genera varios planes OMPL y escoge el mejor según costo de cable."""

        best_plan = None
        best_cost = None
        best_deltas = None

        self.get_logger().info(
            f"Iniciando best-of-{attempts} para pre-grasp con protección de cable."
        )

        for i in range(attempts):
            self.get_logger().info(f"Intento OMPL candidato {i + 1}/{attempts}")

            plan_result = self.arm_plan_to_pose_raw(
                arm=arm,
                pose=pose,
                pose_link=pose_link,
                orientation_tolerance=orientation_tolerance,
                position_tolerance=position_tolerance,
            )

            if plan_result is None:
                self.get_logger().warn(f"  Intento {i + 1}: PLAN_FAILED")
                continue

            cost, deltas = self._trajectory_cable_cost(plan_result.trajectory)

            self.get_logger().info(
                f"  Intento {i + 1}: costo cable={cost:.3f}"
            )
            self._log_cable_deltas(deltas)

            if best_plan is None or cost < best_cost:
                best_plan = plan_result
                best_cost = cost
                best_deltas = deltas

        if best_plan is None:
            self.get_logger().error(
                "Ningún intento OMPL produjo trayectoria válida."
            )
            return False, "PLAN_FAILED", None

        self.get_logger().info(
            f"Mejor plan seleccionado con costo cable={best_cost:.3f}"
        )
        self._log_cable_deltas(best_deltas, prefix="Deltas del mejor plan:")

        # Ahora sí aplicamos límites duros.
        if not self._trajectory_cable_motion_ok(best_plan.trajectory):
            self.get_logger().error(
                "El mejor plan encontrado sigue siendo riesgoso para el cable."
            )
            return False, "CABLE_MOTION_TOO_LARGE", best_plan

        self.get_logger().info("PLAN BEST-OF-N OK ✓")

        if execute:
            exec_result = self.execute_trajectory(
                best_plan.trajectory,
                velocity_scaling=velocity_scaling,
                acceleration_scaling=acceleration_scaling,
            )
            return True, str(exec_result), best_plan

        return True, "BEST_OF_N_PLAN_SUCCEEDED", best_plan


    def plan_and_maybe_execute(
        self,
        planning_component: PlanningComponent,
        execute: bool = False,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
    ):
        """Planifica y, si execute=True, ejecuta."""

        plan_result = self.plan(planning_component)

        if not plan_result:
            self.get_logger().error(
                "PLAN FALLÓ. Puede ser por colisión, goal inalcanzable o timeout."
            )
            return False, "PLAN_FAILED", None

        self.get_logger().info(
            "PLAN OK ✓. La trayectoria fue validada por el pipeline de MoveIt."
        )

        if not self._trajectory_cable_motion_ok(plan_result.trajectory):
            self.get_logger().error(
                "PLAN RECHAZADO: aunque MoveIt encontró trayectoria, "
                "el movimiento articular es riesgoso para el cable."
            )
            return False, "CABLE_MOTION_TOO_LARGE", plan_result

        if execute:
            exec_result = self.execute_trajectory(
                plan_result.trajectory,
                velocity_scaling=velocity_scaling,
                acceleration_scaling=acceleration_scaling,
            )
            return True, str(exec_result), plan_result

        return True, "PLAN_SUCCEEDED", plan_result

    # -------------------------------------------------------------------------
    # Movimientos de brazo
    # -------------------------------------------------------------------------

    def arm_go_to_named_pose(
        self,
        arm: str,
        pose_name: str,
        execute: bool = False,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
    ):
        """Planifica o ejecuta hacia una pose nombrada del SRDF."""

        selected_arm = self._get_arm(arm)

        selected_arm.set_start_state_to_current_state()
        selected_arm.set_goal_state(configuration_name=pose_name)

        self.get_logger().info(
            f"Brazo '{arm}' → named pose '{pose_name}' | execute={execute}"
        )

        return self.plan_and_maybe_execute(
            selected_arm,
            execute=execute,
            velocity_scaling=velocity_scaling,
            acceleration_scaling=acceleration_scaling,
        )

    def arm_go_to_pose(
        self,
        arm: str,
        pose: PoseStamped,
        pose_link: Optional[str] = None,
        execute: bool = False,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
    ):
        """Planifica o ejecuta hacia una PoseStamped."""

        selected_arm = self._get_arm(arm)
        link = self._get_pose_link(arm, pose_link)

        selected_arm.set_start_state_to_current_state()
        selected_arm.set_goal_state(
            pose_stamped_msg=pose,
            pose_link=link,
        )

        self.get_logger().info(
            f"Brazo '{arm}' → pose cartesiana usando pose_link='{link}' | "
            f"frame='{pose.header.frame_id}' | execute={execute}"
        )

        return self.plan_and_maybe_execute(
            selected_arm,
            execute=execute,
            velocity_scaling=velocity_scaling,
            acceleration_scaling=acceleration_scaling,
        )

    # -------------------------------------------------------------------------
    # Collision objects
    # -------------------------------------------------------------------------

    def add_collision_box(
        self,
        object_id: str,
        frame_id: str,
        position: Tuple[float, float, float],
        dimensions: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        color: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        """Agrega una caja a la planning scene."""

        collision_object = CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.header.stamp = self.get_clock().now().to_msg()
        collision_object.id = object_id

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [dimensions[0], dimensions[1], dimensions[2]]

        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]

        collision_object.primitives.append(box)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD

        with self.planning_monitor.read_write() as scene:
            if color is not None:
                object_color = ObjectColor()
                object_color.id = object_id
                object_color.color.r = color[0]
                object_color.color.g = color[1]
                object_color.color.b = color[2]
                object_color.color.a = color[3]
                scene.apply_collision_object(collision_object, object_color)
            else:
                scene.apply_collision_object(collision_object)

            scene.current_state.update()

        self._publish_collision_diff(collision_object)

        self.get_logger().info(
            f"Collision box '{object_id}' agregada en frame '{frame_id}'."
        )
        return True

    def add_collision_cylinder(
        self,
        object_id: str,
        frame_id: str,
        position: Tuple[float, float, float],
        height: float,
        radius: float,
        orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> bool:
        """Agrega un cilindro a la planning scene."""

        collision_object = CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.header.stamp = self.get_clock().now().to_msg()
        collision_object.id = object_id

        cylinder = SolidPrimitive()
        cylinder.type = SolidPrimitive.CYLINDER
        cylinder.dimensions = [height, radius]

        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]

        collision_object.primitives.append(cylinder)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD

        with self.planning_monitor.read_write() as scene:
            scene.apply_collision_object(collision_object)
            scene.current_state.update()

        self._publish_collision_diff(collision_object)

        self.get_logger().info(
            f"Collision cylinder '{object_id}' agregado en frame '{frame_id}'."
        )
        return True

    def remove_collision_object(self, object_id: str) -> bool:
        """Remueve un objeto de colisión por ID."""

        collision_object = CollisionObject()
        collision_object.id = object_id
        collision_object.operation = CollisionObject.REMOVE

        with self.planning_monitor.read_write() as scene:
            scene.apply_collision_object(collision_object)
            scene.current_state.update()

        self._publish_collision_diff(collision_object)

        self.get_logger().info(f"Collision object '{object_id}' removido.")
        return True

    def _publish_collision_diff(self, collision_object: CollisionObject):
        """Publica un diff de planning scene."""

        msg = PlanningScene()
        msg.world.collision_objects.append(collision_object)
        msg.is_diff = True
        self.planning_scene_pub.publish(msg)





    def _on_move_group_scene(self, msg: PlanningScene):
        """Relaya SOLO el OctoMap de move_group al PSM local como diff."""
        octo = msg.world.octomap.octomap

        # Ignorar mensajes sin octomap (incluidos los vacíos que el propio
        # commander publica en este mismo topic). Esto evita pisar el mapa.
        if octo.resolution <= 0.0 or len(octo.data) == 0:
            return

        diff = PlanningScene()
        diff.is_diff = True
        diff.world.octomap = msg.world.octomap  # solo el octomap, nada más

        try:
            self.planning_monitor.new_planning_scene_message(diff)
        except Exception as exc:
            self.get_logger().warn(f"No se pudo aplicar OctoMap de move_group: {exc}")
            return

        if not self._octomap_synced_once:
            self._octomap_synced_once = True
            self.get_logger().info(
                f"OctoMap de move_group sincronizado en el PSM local "
                f"(resolution={octo.resolution:.3f}, bytes={len(octo.data)})."
            )





    def clear_local_octomap(self) -> tuple[bool, str]:
        """Intenta limpiar el OctoMap mantenido por este nodo MoveItPy.

        Importante:
        /clear_octomap normalmente limpia el OctoMap de move_group.
        Como este proyecto usa MoveItPy en /ur_dual_commander, necesitamos
        limpiar el mapa local del PlanningSceneMonitor del commander.
        """

        # Opción 1: método expuesto directamente en PlanningSceneMonitor.
        for method_name in ("clear_octomap", "clearOctomap"):
            if hasattr(self.planning_monitor, method_name):
                try:
                    getattr(self.planning_monitor, method_name)()
                    self.get_logger().info(
                        f"OctoMap local limpiado usando planning_monitor.{method_name}()."
                    )
                    return True, f"OctoMap local limpiado con {method_name}()."
                except Exception as exc:
                    self.get_logger().warn(
                        f"Falló planning_monitor.{method_name}(): {exc}"
                    )

        # Opción 2: algunos bindings podrían exponerlo desde la escena.
        try:
            with self.planning_monitor.read_write() as scene:
                for method_name in ("clear_octomap", "clearOctomap"):
                    if hasattr(scene, method_name):
                        try:
                            getattr(scene, method_name)()
                            scene.current_state.update()
                            self.get_logger().info(
                                f"OctoMap local limpiado usando scene.{method_name}()."
                            )
                            return True, f"OctoMap local limpiado con scene.{method_name}()."
                        except Exception as exc:
                            self.get_logger().warn(
                                f"Falló scene.{method_name}(): {exc}"
                            )
        except Exception as exc:
            self.get_logger().warn(
                f"No se pudo abrir planning_monitor.read_write() para limpiar OctoMap: {exc}"
            )

        msg = (
            "No encontré un método Python expuesto para limpiar el OctoMap local. "
            "Como fallback, reinicia ur_dual_command.launch.py para limpiar el mapa del commander."
        )
        self.get_logger().error(msg)
        return False, msg






    # -------------------------------------------------------------------------
    # Planificación cartesiana real
    # -------------------------------------------------------------------------

    def _trajectory_joint_delta_ok(
        self,
        trajectory: RobotTrajectory,
        joint_name: str = "ur_dual_I_wrist_3_joint",
        max_delta_rad: float = 0.35,
    ) -> bool:
        """Revisa que un joint no gire demasiado dentro de la trayectoria.

        Esto es una protección práctica por el cable de la SoftHand.
        0.35 rad equivale aproximadamente a 20 grados.
        """

        traj_msg = trajectory.get_robot_trajectory_msg()
        joint_names = list(traj_msg.joint_trajectory.joint_names)

        if joint_name not in joint_names:
            self.get_logger().warn(
                f"No encontré {joint_name} en la trayectoria. "
                "No puedo validar giro de muñeca."
            )
            return True

        idx = joint_names.index(joint_name)

        positions = [
            point.positions[idx]
            for point in traj_msg.joint_trajectory.points
            if len(point.positions) > idx
        ]

        if not positions:
            self.get_logger().warn(
                f"No hay posiciones para {joint_name}. "
                "No puedo validar giro de muñeca."
            )
            return True

        delta = max(positions) - min(positions)

        self.get_logger().info(
            f"Chequeo cable/muñeca: {joint_name} delta={delta:.3f} rad "
            f"({delta * 57.2958:.1f} deg)"
        )

        if abs(delta) > max_delta_rad:
            self.get_logger().error(
                f"Trayectoria rechazada: {joint_name} gira demasiado "
                f"({delta:.3f} rad > {max_delta_rad:.3f} rad)."
            )
            return False

        return True

    def cartesian_plan_to_pose(
        self,
        arm: str,
        pose: PoseStamped,
        pose_link: Optional[str] = None,
        max_step: float = 0.005,
        jump_threshold: float = 0.0,
        min_fraction: float = 0.95,
        timeout_s: float = 10.0,
        check_wrist: bool = True,
    ):
        """Planifica una trayectoria cartesiana real hacia una pose.

        Diferencia clave:
        - arm_go_to_pose() usa OMPL hacia un pose goal.
        - cartesian_plan_to_pose() usa /compute_cartesian_path y waypoints.

        Esto se debe usar para descensos/subidas cortas:
        pre-grasp -> grasp
        grasp -> lift
        """

        selected_arm = self._get_arm(arm)
        group_name = selected_arm.planning_group_name
        link = self._get_pose_link(arm, pose_link)

        self.get_logger().info(
            f"Plan cartesiano real: group='{group_name}', link='{link}', "
            f"frame='{pose.header.frame_id}', max_step={max_step}, "
            f"min_fraction={min_fraction}"
        )

        self.get_logger().info(
            "Esperando disponibilidad real de /compute_cartesian_path..."
        )

        service_ready = False

        for attempt in range(1, 16):
            if self.cartesian_plan_client.wait_for_service(timeout_sec=1.0):
                service_ready = True
                break

            visible_services = [
                name
                for name, _types in self.get_service_names_and_types()
                if "compute_cartesian" in name
            ]

            self.get_logger().warn(
                f"/compute_cartesian_path aún no disponible para este nodo "
                f"(intento {attempt}/15). Servicios visibles relacionados: "
                f"{visible_services}"
            )

        if not service_ready:
            self.get_logger().error(
                "El cliente interno no logró conectarse a /compute_cartesian_path. "
                "Si 'ros2 service list' lo muestra, puede ser discovery/daemon o que "
                "el proveedor real del servicio no esté activo para este nodo."
            )
            return False, "CARTESIAN_SERVICE_NOT_AVAILABLE", None

        request = GetCartesianPath.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = pose.header.frame_id
        request.group_name = group_name
        request.link_name = link
        request.max_step = max_step
        request.jump_threshold = jump_threshold
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = 0.05
        request.max_acceleration_scaling_factor = 0.05

        with self.planning_monitor.read_write() as scene:
            current_state = scene.current_state
            assert isinstance(current_state, RobotState)
            current_state.update(True)

            # El start_state ya define desde dónde inicia la trayectoria.
            # NO agregamos la pose actual como waypoint porque puede venir
            # expresada en otro frame y provocar fraction=0.000.
            request.start_state = robotStateToRobotStateMsg(current_state)

            # Solo pasamos el objetivo cartesiano.
            # El frame de este waypoint es request.header.frame_id.
            request.waypoints = [pose.pose]

        future = self.cartesian_plan_client.call_async(request)

        start_time = time.monotonic()

        while rclpy.ok() and not future.done():
            if time.monotonic() - start_time > timeout_s:
                self.get_logger().error(
                    f"Timeout esperando respuesta de /compute_cartesian_path "
                    f"({timeout_s}s)."
                )
                return False, "CARTESIAN_TIMEOUT", None

            time.sleep(0.05)

        response = future.result()

        if response is None:
            self.get_logger().error("Respuesta vacía de /compute_cartesian_path.")
            return False, "CARTESIAN_EMPTY_RESPONSE", None

        self.get_logger().info(
            f"Cartesian fraction={response.fraction:.3f}"
        )

        if response.fraction < min_fraction:
            self.get_logger().error(
                f"Trayectoria cartesiana incompleta: fraction={response.fraction:.3f} "
                f"< min_fraction={min_fraction:.3f}."
            )
            return False, "CARTESIAN_FRACTION_TOO_LOW", None

        robot_trajectory = RobotTrajectory(self.robot_model)
        robot_trajectory.set_robot_trajectory_msg(current_state, response.solution)
        robot_trajectory.joint_model_group_name = group_name

        if check_wrist:
            if not self._trajectory_joint_delta_ok(
                robot_trajectory,
                joint_name="ur_dual_I_wrist_3_joint",
                max_delta_rad=0.35,
            ):
                return False, "WRIST_ROTATION_TOO_LARGE", None

        plan_like_result = SimpleNamespace(trajectory=robot_trajectory)

        self.get_logger().info("PLAN CARTESIANO OK ✓")
        return True, "CARTESIAN_PLAN_SUCCEEDED", plan_like_result


    def cartesian_plan_through_poses(
        self,
        arm: str,
        poses: list[PoseStamped],
        pose_link: Optional[str] = None,
        max_step: float = 0.005,
        jump_threshold: float = 0.0,
        min_fraction: float = 0.95,
        timeout_s: float = 15.0,
        check_cable: bool = True,
        start_joint_state_msg: Optional[JointState] = None,
    ):
        """Planifica una trayectoria cartesiana real pasando por varios waypoints.

        Esto sirve para:
        - mover primero en X/Y a una altura segura
        - luego bajar en Z hacia pre-grasp
        - evitar piruetas de muñeca típicas de OMPL
        """

        if not poses:
            self.get_logger().error("No se recibieron waypoints cartesianos.")
            return False, "NO_CARTESIAN_WAYPOINTS", None

        selected_arm = self._get_arm(arm)
        group_name = selected_arm.planning_group_name
        link = self._get_pose_link(arm, pose_link)

        frame_id = poses[0].header.frame_id

        for idx, pose in enumerate(poses):
            if pose.header.frame_id != frame_id:
                self.get_logger().error(
                    f"Waypoint {idx} tiene frame '{pose.header.frame_id}', "
                    f"pero se esperaba '{frame_id}'."
                )
                return False, "WAYPOINT_FRAME_MISMATCH", None

        self.get_logger().info(
            f"Plan cartesiano por waypoints: group='{group_name}', "
            f"link='{link}', frame='{frame_id}', "
            f"waypoints={len(poses)}, max_step={max_step}, "
            f"min_fraction={min_fraction}"
        )

        self.get_logger().info(
            "Esperando disponibilidad real de /compute_cartesian_path..."
        )

        service_ready = False

        for attempt in range(1, 16):
            if self.cartesian_plan_client.wait_for_service(timeout_sec=1.0):
                service_ready = True
                break

            visible_services = [
                name
                for name, _types in self.get_service_names_and_types()
                if "compute_cartesian" in name
            ]

            self.get_logger().warn(
                f"/compute_cartesian_path aún no disponible para este nodo "
                f"(intento {attempt}/15). Servicios visibles relacionados: "
                f"{visible_services}"
            )

        if not service_ready:
            self.get_logger().error(
                "El cliente interno no logró conectarse a /compute_cartesian_path."
            )
            return False, "CARTESIAN_SERVICE_NOT_AVAILABLE", None

        request = GetCartesianPath.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = frame_id
        request.group_name = group_name
        request.link_name = link
        request.max_step = max_step
        request.jump_threshold = jump_threshold
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = 0.05
        request.max_acceleration_scaling_factor = 0.05

        with self.planning_monitor.read_write() as scene:
            current_state = scene.current_state
            assert isinstance(current_state, RobotState)
            current_state.update(True)

            if start_joint_state_msg is not None:
                start_state_msg = RobotStateMsg()
                start_state_msg.joint_state = start_joint_state_msg
                start_state_msg.is_diff = False
                request.start_state = start_state_msg

                self.get_logger().info(
                    "Usando /joint_states real como start_state para Cartesian Path."
                )
            else:
                request.start_state = robotStateToRobotStateMsg(current_state)

                self.get_logger().warn(
                    "Usando scene.current_state como start_state. "
                    "Si el robot fue movido manualmente, esto puede estar desfasado."
                )

            request.waypoints = [pose.pose for pose in poses]

        future = self.cartesian_plan_client.call_async(request)

        start_time = time.monotonic()

        while rclpy.ok() and not future.done():
            if time.monotonic() - start_time > timeout_s:
                self.get_logger().error(
                    f"Timeout esperando respuesta de /compute_cartesian_path "
                    f"({timeout_s}s)."
                )
                return False, "CARTESIAN_TIMEOUT", None

            time.sleep(0.05)

        response = future.result()

        if response is None:
            self.get_logger().error("Respuesta vacía de /compute_cartesian_path.")
            return False, "CARTESIAN_EMPTY_RESPONSE", None

        self.get_logger().info(f"Cartesian fraction={response.fraction:.3f}")

        if response.fraction < min_fraction:
            self.get_logger().error(
                f"Trayectoria cartesiana incompleta: fraction={response.fraction:.3f} "
                f"< min_fraction={min_fraction:.3f}."
            )
            return False, "CARTESIAN_FRACTION_TOO_LOW", None

        robot_trajectory = RobotTrajectory(self.robot_model)
        robot_trajectory.set_robot_trajectory_msg(current_state, response.solution)
        robot_trajectory.joint_model_group_name = group_name

        if check_cable:
            if not self._trajectory_cable_motion_ok(robot_trajectory):
                return False, "CABLE_MOTION_TOO_LARGE", SimpleNamespace(
                    trajectory=robot_trajectory
                )

        plan_like_result = SimpleNamespace(trajectory=robot_trajectory)

        self.get_logger().info("PLAN CARTESIANO POR WAYPOINTS OK ✓")
        return True, "CARTESIAN_WAYPOINT_PLAN_SUCCEEDED", plan_like_result


    def arm_go_to_pose_cartesian(
        self,
        arm: str,
        pose: PoseStamped,
        pose_link: Optional[str] = None,
        execute: bool = False,
        velocity_scaling: float = 0.05,
        acceleration_scaling: float = 0.05,
    ):
        """Planifica, y opcionalmente ejecuta, una trayectoria cartesiana real."""

        success, status, plan_result = self.cartesian_plan_to_pose(
            arm=arm,
            pose=pose,
            pose_link=pose_link,
            max_step=0.005,
            jump_threshold=0.0,
            min_fraction=0.95,
            timeout_s=10.0,
            check_wrist=True,
        )

        if not success or plan_result is None:
            return success, status, None

        if execute:
            exec_result = self.execute_trajectory(
                plan_result.trajectory,
                velocity_scaling=velocity_scaling,
                acceleration_scaling=acceleration_scaling,
            )
            return True, str(exec_result), plan_result

        return True, status, plan_result


    def shutdown(self):
        """Cierre explícito del wrapper."""

        try:
            self.robot.shutdown()
        except Exception:
            pass