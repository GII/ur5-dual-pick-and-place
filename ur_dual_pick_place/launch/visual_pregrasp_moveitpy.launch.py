import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # Carga la configuración completa de MoveIt del paquete ur_dual_moveit_config.
    # Esto es lo que faltaba cuando se ejecutaba con ros2 run.
    ur_moveit_config_path = get_package_share_directory("ur_dual_moveit_config")
    moveit_py_config_path = os.path.join(
        ur_moveit_config_path,
        "config",
        "moveit_py_ur_dual.yaml",
    )
    moveit_config = (
        MoveItConfigsBuilder("ur_dual", package_name="ur_dual_moveit_config")
        .robot_description_semantic(file_path="config/ur_dual.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl"],
        )
        .moveit_cpp(file_path=moveit_py_config_path)
        .to_moveit_configs()
    )

    return LaunchDescription([
        Node(
            package="ur_dual_pick_place",
            executable="visual_pregrasp_moveitpy",
            name="visual_pregrasp_moveitpy",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "target_class": "cubo",
                    "pose_link": "qbhand2m1_palm_link",
                    "palm_z_offset": 0.0,
                    "approach_height": 0.30,
                    "orientation_roll": 0.0,
                    "orientation_pitch": 0.0,
                    "orientation_yaw": 0.0,
                    "plan_only": True,
                },
            ],
        )
    ])
