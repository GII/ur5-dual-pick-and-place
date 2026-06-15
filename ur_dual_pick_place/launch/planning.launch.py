from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ur_dual_moveit_config"), "/launch/move_group.launch.py"]
        ),
    )

    commander = TimerAction(
        period=6.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [FindPackageShare("ur_dual_pick_place"), "/launch/ur_dual_command.launch.py"]
            ),
        )],
    )
    bridge = TimerAction(
        period=6.0,
        actions=[Node(
            package="ur_dual_pick_place",
            executable="object_pose_bridge",
            name="object_pose_bridge",
            output="screen",
            parameters=[{"target_class": "", "min_confidence": 0.50}],
        )],
    )
    return LaunchDescription([move_group, commander, bridge])
