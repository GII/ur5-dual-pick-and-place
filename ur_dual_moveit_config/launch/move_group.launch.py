"""
move_group.launch.py

Launch del nodo move_group con OctoMap habilitado y, MUY IMPORTANTE,
con publicación del planning scene completo habilitada para que otros
nodos (como el commander MoveItPy) puedan ver el OctoMap.

Por defecto, generate_move_group_launch() crea move_group con
publish_planning_scene=False, así que el OctoMap se mantiene dentro
de move_group y no se distribuye. Acá lo habilitamos explícitamente.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("ur_dual", package_name="ur_dual_moveit_config")
        .sensors_3d(file_path="config/octomap_sensors.yaml")
        .to_moveit_configs()
    )

    # ============================================================
    # Parámetros adicionales para que move_group PUBLIQUE el
    # planning scene completo (incluyendo OctoMap) a los suscriptores.
    # Sin esto, el commander MoveItPy mantiene su escena local vacía
    # y el OctoMap nunca llega a sus planificaciones.
    # ============================================================
    move_group_capabilities_params = {
        # CRÍTICO: publicar la escena completa, no solo diffs.
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        # Frecuencia de publicación del planning scene (Hz).
        "planning_scene_monitor_options.publish_planning_scene_hz": 25.0,
        # Habilitar las capabilities estándar.
        "capabilities": "",
        "disable_capabilities": "",
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            move_group_capabilities_params,
        ],
    )

    return LaunchDescription([move_group_node])