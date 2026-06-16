"""
ur_dual_command.launch.py

Brings up the manipulation back-end:
  - octomap_input_filter : category-aware depth filter (removes the target
                           object's voxels so the OctoMap never blocks the grasp)
  - ur_dual_commander    : MoveItPy commander exposing the pick-and-place services

The commander is loaded with the MoveIt config + motion_planning.yaml so its
MoveItPy planning pipelines come up correctly.
"""

import os

from launch import LaunchDescription
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

    depth_crop_node = Node(
        package="ur_dual_pick_place",
        executable="depth_crop_to_cloud",
        name="depth_crop_to_cloud",
        output="screen",
        parameters=[
            {
                "depth_topic": "/oak_cam/oak/stereo/image_raw",
                "camera_info_topic": "/oak_cam/oak/stereo/camera_info",
                "output_topic": "/ur_dual/cropped_obstacle_cloud",
                "base_frame": "ur_dual_I_base_link",

                "subsample": 4,

                "min_depth_m": 0.45,
                "max_depth_m": 1.25,

                "min_x": -0.75,
                "max_x": 0.25,
                "min_y": 0.35,
                "max_y": 1.10,
                "min_z": 0.02,
                "max_z": 0.75,

                "publish_period_s": 0.2,
                "min_points_before_publish": 50,
            }
        ],
    )



    obstacle_clusterer_node = Node(
        package="ur_dual_pick_place",
        executable="obstacle_clusterer",
        name="obstacle_clusterer",
        output="screen",
        parameters=[
            {
                "depth_topic": "/oak_cam/oak/stereo/image_raw",
                "camera_info_topic": "/oak_cam/oak/stereo/camera_info",
                "base_frame": "ur_dual_I_base_link",
                "object_pose_topic": "/ur_dual/object_pose",
                # Workspace ROI (frame base)
                "min_x": -0.75, "max_x": 0.25,
                "min_y": 0.30,  "max_y": 1.10,
                "min_z": 0.005, "max_z": 0.60,
                # Filtro profundidad cámara
                "min_depth_m": 0.20, "max_depth_m": 1.50,
                "depth_subsample": 3,
                # RANSAC mesa
                "ransac_distance_threshold": 0.012,
                "ransac_n_iterations": 500,
                "expected_table_normal_z_min": 0.85,
                # Objeto target
                "object_exclusion_radius": 0.10,
                "object_pose_max_age_s": 3.0,
                # DBSCAN
                "dbscan_eps": 0.03,
                "dbscan_min_samples": 30,
                "min_cluster_size": 80,
                "max_clusters": 8,
                # Padding alrededor del bbox real
                "bbox_padding": 0.02,
            }
        ],
    )

    octomap_filter_node = Node(
        package="ur_dual_pick_place",
        executable="octomap_input_filter",
        name="octomap_input_filter",
        output="screen",
        parameters=[
            {
                "depth_topic_in": "/oak_cam/oak/stereo/image_raw",
                "caminfo_topic_in": "/oak_cam/oak/stereo/camera_info",
                "depth_topic_out": "/oak_cam/oak/stereo/image_raw_filtered",
                "caminfo_topic_out": "/oak_cam/oak/stereo/camera_info_filtered",
                "base_frame": "ur_dual_I_base_link",
                "filter_margin": 0.04,
                "object_pose_max_age_s": 5.0,
                "clear_value_mm": 0,
            }
        ],
    )


    commander_config = moveit_config.to_dict()



    commander_node = Node(
        package="ur_dual_pick_place",
        executable="ur_dual_command_services",
        name="ur_dual_commander",
        output="screen",
        parameters=[commander_config],
    )

    return LaunchDescription([octomap_filter_node, obstacle_clusterer_node, commander_node])

