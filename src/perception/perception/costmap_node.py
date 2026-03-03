"""
Road-line → costmap node  (Option A – Python)

Pipeline
--------
1.  Subscribe to a raw camera image  (/camera/image_raw)
2.  (Optional) Undistort using camera_info intrinsics
3.  Apply a homography to produce a bird's-eye / top-down view
4.  Detect white road lines via adaptive-threshold + morphology
5.  Map the binary mask → lethal (100) / free (0) OccupancyGrid costs
6.  Publish on /perception/road_costmap  (nav_msgs/OccupancyGrid)

Nav2 can ingest this grid through a costmap_2d "static layer" or
"voxel layer" pointed at the same topic.

All tuneable values are exposed as ROS 2 parameters so they can be
set from a launch file or YAML without touching code.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


class RoadLineCostmapNode(Node):
    """Detect white road lines and publish them as a Nav2-compatible costmap."""

    # -------------------------------------------------------------------
    #  Construction / parameter declaration
    # -------------------------------------------------------------------
    def __init__(self):
        super().__init__('road_line_costmap_node')

        # ---- declare parameters (all overridable) ---------------------
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('costmap_topic', '/perception/road_costmap')
        self.declare_parameter('costmap_frame', 'base_link')

        # Homography – 3×3 flattened row-major  (identity = no warp)
        self.declare_parameter(
            'homography',[-0.21489038,-0.63978727,573.46916994,-0.01868842,-1.22596043,788.34488373,-0.00002152,-0.00169354,1.00000000]
        )
        # Bird-eye output size
        self.declare_parameter('bev_width', 800)
        self.declare_parameter('bev_height', 600)

        # White-line detection
        self.declare_parameter('adaptive_block_size', 51)
        self.declare_parameter('adaptive_c', -25)
        self.declare_parameter('morph_kernel_size', 3)
        self.declare_parameter('min_brightness', 180)

        # Costmap geometry  (metres / cell)
        self.declare_parameter('resolution', 0.01)
        # Origin of the grid in the costmap_frame (metres)
        self.declare_parameter('origin_x', -1 * self.get_parameter('bev_width').get_parameter_value().integer_value * self.get_parameter('resolution').get_parameter_value().double_value / 2)
        self.declare_parameter('origin_y', -1 * self.get_parameter('bev_height').get_parameter_value().integer_value * self.get_parameter('resolution').get_parameter_value().double_value / 2)

        # Undistortion toggle
        self.declare_parameter('undistort', True)

        # Show OpenCV debug windows (BEV + mask)
        self.declare_parameter('show_debug', True)

        # ---- read parameters ------------------------------------------
        self._read_params()

        # ---- state variables ------------------------------------------
        self._bridge = CvBridge()
        self._camera_matrix = None
        self._dist_coeffs = None

        # ---- QoS – sensor-data compatible ----------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- subscribers ----------------------------------------------
        self.create_subscription(
            Image,
            self._image_topic,
            self._image_cb,
            sensor_qos,
        )
        self.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            self._camera_info_cb,
            sensor_qos,
        )

        # ---- publisher ------------------------------------------------
        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._costmap_pub = self.create_publisher(
            OccupancyGrid,
            self._costmap_topic,
            costmap_qos,
        )

        self.get_logger().info(
            f'Road-line costmap node started  '
            f'[img={self._image_topic}  '
            f'costmap={self._costmap_topic}  '
            f'frame={self._costmap_frame}]'
        )

    # -------------------------------------------------------------------
    #  Parameter helpers
    # -------------------------------------------------------------------
    def _read_params(self):
        self._image_topic = (
            self.get_parameter('image_topic').get_parameter_value().string_value
        )
        self._camera_info_topic = (
            self.get_parameter('camera_info_topic')
            .get_parameter_value()
            .string_value
        )
        self._costmap_topic = (
            self.get_parameter('costmap_topic')
            .get_parameter_value()
            .string_value
        )
        self._costmap_frame = (
            self.get_parameter('costmap_frame')
            .get_parameter_value()
            .string_value
        )
        h_flat = (
            self.get_parameter('homography')
            .get_parameter_value()
            .double_array_value
        )
        self._homography = np.array(h_flat, dtype=np.float64).reshape(3, 3)

        self._bev_w = (
            self.get_parameter('bev_width')
            .get_parameter_value()
            .integer_value
        )
        self._bev_h = (
            self.get_parameter('bev_height')
            .get_parameter_value()
            .integer_value
        )
        self._block_size = (
            self.get_parameter('adaptive_block_size')
            .get_parameter_value()
            .integer_value
        )
        self._adaptive_c = (
            self.get_parameter('adaptive_c')
            .get_parameter_value()
            .integer_value
        )
        self._morph_k = (
            self.get_parameter('morph_kernel_size')
            .get_parameter_value()
            .integer_value
        )
        self._min_bright = (
            self.get_parameter('min_brightness')
            .get_parameter_value()
            .integer_value
        )
        self._resolution = (
            self.get_parameter('resolution')
            .get_parameter_value()
            .double_value
        )
        self._origin_x = (
            self.get_parameter('origin_x')
            .get_parameter_value()
            .double_value
        )
        self._origin_y = (
            self.get_parameter('origin_y')
            .get_parameter_value()
            .double_value
        )
        self._undistort = (
            self.get_parameter('undistort')
            .get_parameter_value()
            .bool_value
        )
        self._show_debug = (
            self.get_parameter('show_debug')
            .get_parameter_value()
            .bool_value
        )

    # -------------------------------------------------------------------
    #  Callbacks
    # -------------------------------------------------------------------
    def _camera_info_cb(self, msg: CameraInfo):
        """Cache intrinsics for undistortion (received once is enough)."""
        if self._camera_matrix is not None:
            return  # already have them
        self._camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.get_logger().info('Camera intrinsics received – undistortion enabled.')

    def _image_cb(self, msg: Image):
        """Main processing pipeline – runs on every incoming frame."""
        # --- 1. Convert ROS image → OpenCV BGR ---------------------------
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'cv_bridge error: {exc}')
            return

        # --- 2. Undistort (if intrinsics available) -----------------------
        if self._undistort and self._camera_matrix is not None:
            frame = cv2.undistort(
                frame, self._camera_matrix, self._dist_coeffs
            )

        # --- 3. Homography → bird's-eye view -----------------------------
        bev = cv2.warpPerspective(
            frame,
            self._homography,
            (self._bev_w, self._bev_h),
            flags=cv2.INTER_LINEAR,
        )

        # --- 4. White-line detection --------------------------------------
        mask = self._detect_white_lines(bev)

        # --- 5. Build and publish OccupancyGrid ---------------------------
        grid = self._mask_to_occupancy_grid(mask, msg.header.stamp)
        self._costmap_pub.publish(grid)

        # --- 6. Debug visualisation (OpenCV windows) ----------------------
        if self._show_debug:
            cv2.imshow('BEV (Bird-Eye View)', bev)
            cv2.imshow('Line Mask', mask)
            cv2.waitKey(1)

    # -------------------------------------------------------------------
    #  Image-processing helpers
    # -------------------------------------------------------------------
    def _detect_white_lines(self, bev_bgr: np.ndarray) -> np.ndarray:
        """
        Return a binary mask (0 / 255) of detected white road lines.

        Strategy
        --------
        1. Convert to grayscale.
        2. Brightness gate – only keep pixels above *min_brightness*.
        3. Adaptive threshold – highlights locally-bright pixels (lines)
           against the surrounding road surface.
        4. AND the two masks so only bright + locally-bright pixels survive.
        5. Morphological close to fill small gaps in dashed lines.
        """
        gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)

        # Brightness gate
        _, bright_mask = cv2.threshold(
            gray, self._min_bright, 255, cv2.THRESH_BINARY
        )

        # Ensure block size is odd and >= 3
        bs = max(3, self._block_size | 1)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            self._adaptive_c,
        )

        # Combine
        mask = cv2.bitwise_and(bright_mask, adaptive)

        # Morphological close (fills small gaps in dashed lines)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self._morph_k, self._morph_k)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _mask_to_occupancy_grid(
        self, mask: np.ndarray, stamp: Time
    ) -> OccupancyGrid:
        """
        Convert a binary image mask to a nav_msgs/OccupancyGrid.

        - White (255) → lethal cost   100
        - Black (0)   → free-space      0
        """
        grid = OccupancyGrid()

        # Header
        grid.header = Header()
        grid.header.stamp = stamp
        grid.header.frame_id = self._costmap_frame

        # Map metadata
        grid.info.resolution = self._resolution
        grid.info.width = mask.shape[1]
        grid.info.height = mask.shape[0]

        origin = Pose()
        origin.position.x = self._origin_x
        origin.position.y = self._origin_y
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        grid.info.origin = origin

        # Data – OccupancyGrid uses row-major, values in [-1, 100]
        # Flip vertically so row 0 is the "bottom" of the bird-eye image
        # (closest to the robot), matching ROS convention.
        flipped = np.flipud(mask)
        cost = np.where(flipped > 0, 0, 100).astype(np.int8)
        grid.data = cost.ravel().tolist()

        return grid


# -----------------------------------------------------------------------
#  Entry point
# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RoadLineCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

