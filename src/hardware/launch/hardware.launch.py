from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_rplidar = get_package_share_directory('rplidar_ros')
    rplidar_launch = os.path.join(pkg_rplidar, 'launch', 'rplidar_a3_launch.py')

    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rplidar_launch),
        launch_arguments={
            'frame_id': 'laser'
        }.items()
    )

    pkg_realsense = get_package_share_directory('realsense2_camera')
    realsense_launch = os.path.join(pkg_realsense, 'launch', 'rs_launch.py')

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        launch_arguments={
            'enable_color': 'false',
            'enable_depth': 'false',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2'
        }.items()
    )

    return LaunchDescription([
        lidar,
        realsense
    ])