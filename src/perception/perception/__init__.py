#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge


class LaneHomographyNode(Node):

    def __init__(self):
        super().__init__('lane_homography_node')

        self.bridge = CvBridge()

        # ---------------- PARAMETERS ----------------
        self.declare_parameter("resolution", 0.05)  # meters per pixel
        self.declare_parameter("width", 600)
        self.declare_parameter("height", 600)

        self.resolution = self.get_parameter("resolution").value
        self.width = self.get_parameter("width").value
        self.height = self.get_parameter("height").value

        # ---------------- CAMERA CALIBRATION ----------------
        # Replace with YOUR calibration data
        self.camera_matrix = np.array([
            [700, 0, 640],
            [0, 700, 360],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros(5)

        # ---------------- HOMOGRAPHY POINTS ----------------
        self.src_pts = np.float32([
            [400, 700],   # bottom-left
            [880, 700],   # bottom-right
            [720, 420],   # top-right
            [560, 420]    # top-left
        ])

        self.dst_pts = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width, 0],
            [0, 0]
        ])

        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

        # ---------------- ROS INTERFACES ----------------
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            '/lane_costmap',
            10
        )

        self.get_logger().info("Lane Homography Node Started")

    # ----------------------------------------------------

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 1. Undistort
        undistorted = cv2.undistort(
            frame,
            self.camera_matrix,
            self.dist_coeffs
        )

        # 2. Perspective transform (Top-down)
        top_down = cv2.warpPerspective(
            undistorted,
            self.H,
            (self.width, self.height)
        )

        # 3. Convert to grayscale
        gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

        # 4. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 5. Threshold white lines
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            -5
        )

        # 6. Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 7. Convert to OccupancyGrid
        costmap_msg = OccupancyGrid()
        costmap_msg.header.stamp = self.get_clock().now().to_msg()
        costmap_msg.header.frame_id = "map"

        costmap_msg.info.resolution = self.resolution
        costmap_msg.info.width = self.width
        costmap_msg.info.height = self.height

        # Center the costmap in front of robot
        costmap_msg.info.origin.position.x = 0.0
        costmap_msg.info.origin.position.y = 0.0
        costmap_msg.info.origin.orientation.w = 1.0

        # White lines = obstacle (100), road = free (0)
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        grid[binary == 255] = 100

        costmap_msg.data = grid.flatten().tolist()

        self.costmap_pub.publish(costmap_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneHomographyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()