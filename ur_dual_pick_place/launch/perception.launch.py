import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("depthai_ros_driver"), "/launch/camera.launch.py"]
        ),
        launch_arguments={"namespace": "oak_cam", "pointcloud.enable": "true"}.items(),
    )
    handeye = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ur_dual_calibration"), "/launch/publish_handeye.launch.py"]
        ),
    )
    stereo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("stereo_location"), "/launch/ur5_perception.launch.py"]
        ),
    )

    apply_tuning = TimerAction(
        period=10.0,
        actions=[ExecuteProcess(
            cmd=["bash", os.path.expanduser("~/ws_daniel/apply_oak_depth_tuned.sh")],
            output="screen",
        )],
    )
    return LaunchDescription([camera, handeye, stereo, apply_tuning])
