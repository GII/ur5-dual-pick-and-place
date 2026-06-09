import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ur_dual_pick_place'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='daniel@todo.todo',
    description='Dual-arm UR5 + SoftHand pick and place integration package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_transformer = ur_dual_pick_place.pose_transformer_node:main',
            'visual_pregrasp_moveitpy = ur_dual_pick_place.visual_pregrasp_moveitpy_node:main',
            'move_to_home_right = ur_dual_pick_place.move_to_home_right_node:main',
            'pregrasp_step1_hello = ur_dual_pick_place.pregrasp_step1_hello:main',
            'pregrasp_step2_named_pose = ur_dual_pick_place.pregrasp_step2_named_pose:main',
            'pregrasp_step3_fixed_pose = ur_dual_pick_place.pregrasp_step3_fixed_pose:main',
            'pregrasp_step3b_measured_pose = ur_dual_pick_place.pregrasp_step3b_measured_pose:main',
            'ur_dual_home_test = ur_dual_pick_place.ur_dual_home_test:main',
            'ur_dual_command_services = ur_dual_pick_place.ur_dual_command_services:main',
            'object_pose_bridge = ur_dual_pick_place.object_pose_bridge:main',
            'depth_crop_to_cloud = ur_dual_pick_place.depth_crop_to_cloud:main',
            'obstacle_clusterer = ur_dual_pick_place.obstacle_clusterer:main',
            'octomap_input_filter = ur_dual_pick_place.octomap_input_filter:main',
	    'visual_pregrasp = ur_dual_pick_place.visual_pregrasp_node:main',
        ],
    },
)
