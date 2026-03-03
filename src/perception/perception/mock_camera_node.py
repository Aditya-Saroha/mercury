"""
Mock camera publisher for testing the road-line costmap pipeline.

Reads a video file or a directory of JPEG/PNG images and publishes
frames on /camera/image_raw (Image) + /camera/camera_info (CameraInfo).

The `video_path` parameter is **required** — it must point to either
a video file (.mp4, .avi, .mkv, …) or a folder of images.
"""

import glob
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo


class MockCameraNode(Node):
    def __init__(self):
        super().__init__('mock_camera_node')

        self.declare_parameter('video_path', '')   # path to video OR image dir
        self.declare_parameter('loop', True)        # restart when video ends
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 60.0)

        self._video_path = (
            self.get_parameter('video_path')
            .get_parameter_value()
            .string_value
        )
        self._loop = (
            self.get_parameter('loop')
            .get_parameter_value()
            .bool_value
        )
        self._w = self.get_parameter('width').get_parameter_value().integer_value
        self._h = self.get_parameter('height').get_parameter_value().integer_value
        fps = self.get_parameter('fps').get_parameter_value().double_value

        self._bridge = CvBridge()
        self._frame_id = 0

        # ---- source setup ------------------------------------------------
        self._cap = None          # cv2.VideoCapture (for video files)
        self._image_files = []    # sorted list of image paths (for dir mode)
        self._img_index = 0
        self._last_frame = None   # cache last good frame to avoid flicker

        if not self._video_path:
            self.get_logger().fatal(
                'video_path parameter is required. '
                'Usage: ros2 run perception mock_camera_node '
                '--ros-args -p video_path:=/path/to/video.mp4'
            )
            raise SystemExit(1)

        self._init_source(self._video_path)

        # ---- publishers ---------------------------------------------------
        self._img_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self._info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        self._timer = self.create_timer(1.0 / fps, self._publish)
        self.get_logger().info(
            f'Mock camera started: {self._w}x{self._h} @ {fps} Hz  '
            f'[source={self._video_path}  loop={self._loop}]'
        )

    # ------------------------------------------------------------------
    #  Source initialisation
    # ------------------------------------------------------------------
    def _init_source(self, path: str):
        """Open a video file or scan a directory for JPEG/PNG images."""
        if os.path.isdir(path):
            exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(path, ext)))
                files.extend(glob.glob(os.path.join(path, ext.upper())))
            self._image_files = sorted(set(files))
            if not self._image_files:
                self.get_logger().fatal(f'No images found in {path}')
                raise SystemExit(1)
            self.get_logger().info(
                f'Image-directory mode: {len(self._image_files)} frames in {path}'
            )
        else:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                self.get_logger().fatal(f'Cannot open video: {path}')
                raise SystemExit(1)
            self.get_logger().info(f'Video-file mode: {path}')

    # ------------------------------------------------------------------
    def _publish(self):
        stamp = self.get_clock().now().to_msg()

        frame = self._next_frame()
        if frame is None:
            return  # nothing to publish

        img_msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = 'camera_link'
        self._img_pub.publish(img_msg)

        h, w = frame.shape[:2]
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = 'camera_link'
        info.width = w
        info.height = h
        fx = fy = float(w)
        cx, cy = w / 2.0, h / 2.0
        info.k = [fx, 0.0, cx,
                   0.0, fy, cy,
                   0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.r = [1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0,
                   0.0, fy, cy, 0.0,
                   0.0, 0.0, 1.0, 0.0]
        self._info_pub.publish(info)

        self._frame_id += 1

    # ------------------------------------------------------------------
    #  Frame sources
    # ------------------------------------------------------------------
    def _next_frame(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if unavailable."""
        # 1. Video file
        if self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                if not ret:
                    return self._last_frame  # reuse last good frame
            frame = cv2.resize(frame, (self._w, self._h))
            self._last_frame = frame
            return frame

        # 2. Image directory
        if self._image_files:
            path = self._image_files[self._img_index]
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            self._img_index += 1
            if self._img_index >= len(self._image_files):
                if self._loop:
                    self._img_index = 0
                else:
                    self._img_index = len(self._image_files) - 1
            if frame is None:
                self.get_logger().warn(f'Failed to read {path}')
                return self._last_frame
            frame = cv2.resize(frame, (self._w, self._h))
            self._last_frame = frame
            return frame

        return None


def main(args=None):
    rclpy.init(args=args)
    try:
        node = MockCameraNode()
    except SystemExit:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

