from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ur_dual_pick_place',
            executable='pose_transformer',
            name='pose_transformer',
            output='screen',
            parameters=[{
                'input_topic': '/object_tracker/detections',
                'output_topic': '/object_tracker/detections_base',
                'target_frame': 'ur_dual_I_base_link',
                'tf_timeout_s': 0.2,
            }],
        ),
    ])
