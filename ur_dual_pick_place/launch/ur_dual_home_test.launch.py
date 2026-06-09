# -----------------------------------------------------------------------------
# ur_dual_home_test.launch.py
#
# Lanza:
#   1. RViz con la configuración completa de MoveIt.
#   2. El nodo ur_dual_home_test después de unos segundos.
#
# No lanza move_group.
# MoveItPy usa su propio MoveItCpp interno.
# -----------------------------------------------------------------------------

import os

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    motion_planning_yaml = os.path.join(
        get_package_share_directory("ur_dual_pick_place"),
        "config",
        "motion_planning.yaml",
    )

    moveit_config = (
        MoveItConfigsBuilder("ur_dual", package_name="ur_dual_moveit_config")
        .planning_pipelines(
            pipelines=["ompl"],
            default_planning_pipeline="ompl",
        )
        .moveit_cpp(file_path=motion_planning_yaml)
        .to_moveit_configs()
    )

    rviz_config_file = os.path.join(
        get_package_share_directory("ur_dual_moveit_config"),
        "config",
        "moveit.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[moveit_config.to_dict()],
    )

    # Esperamos unos segundos para que RViz se abra antes de publicar el plan.
    home_test_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="ur_dual_pick_place",
                executable="ur_dual_home_test",
                name="ur_dual_commander",
                output="screen",
                parameters=[moveit_config.to_dict()],
            )
        ],
    )

    return LaunchDescription([
        rviz_node,
        home_test_node,
    ])
