"""
Interactive homography calibration tool.

Usage
-----
    ros2 run perception calibrate_homography --ros-args \
        -p video_path:=/path/to/video.mp4

1.  A window opens showing the first frame (or live feed).
2.  Click **4 points** on the road surface that form a rectangle in
    real life (e.g. the four corners of a lane segment).
    Order: top-left → top-right → bottom-right → bottom-left.
3.  The tool computes the homography, shows the warped BEV preview,
    and prints the 9-element array you can paste straight into a
    launch file or --ros-args.

Press 'r' to reset points.  Press 'q' / ESC to quit.
"""

import cv2
import numpy as np
import sys


# ── globals for mouse callback ────────────────────────────────────────
_points: list = []
_frame = None
_clone = None

BEV_W = 640
BEV_H = 480


def _mouse_cb(event, x, y, flags, param):
    global _points, _clone
    if event == cv2.EVENT_LBUTTONDOWN and len(_points) < 4:
        _points.append((x, y))
        _clone = _frame.copy()
        for i, pt in enumerate(_points):
            cv2.circle(_clone, pt, 6, (0, 255, 0), -1)
            cv2.putText(
                _clone, str(i + 1), (pt[0] + 8, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        if len(_points) >= 2:
            for i in range(len(_points) - 1):
                cv2.line(_clone, _points[i], _points[i + 1], (0, 255, 0), 2)
            if len(_points) == 4:
                cv2.line(_clone, _points[3], _points[0], (0, 255, 0), 2)


def _parse_arg(name: str, default=None):
    """Parse a --ros-args -p name:=value or name=value from sys.argv."""
    for i, arg in enumerate(sys.argv):
        if ':=' in arg and name in arg:
            return arg.split(':=', 1)[1]
        if '=' in arg and name in arg:
            return arg.split('=', 1)[1]
    return default


def main():
    global _frame, _clone, _points, BEV_W, BEV_H

    video_path = _parse_arg('video_path')

    # Also accept plain last argument
    if video_path is None and len(sys.argv) > 1:
        candidate = sys.argv[-1]
        if not candidate.startswith('-'):
            video_path = candidate

    if not video_path:
        print('Usage: calibrate_homography <video_path> [bev_width=W] [bev_height=H]')
        print('   or: ros2 run perception calibrate_homography '
              '--ros-args -p video_path:=/path/to/video.mp4')
        sys.exit(1)

    # Allow overriding BEV size
    bw = _parse_arg('bev_width')
    bh = _parse_arg('bev_height')
    if bw:
        BEV_W = int(bw)
    if bh:
        BEV_H = int(bh)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'ERROR: Cannot open {video_path}')
        sys.exit(1)

    ret, _frame = cap.read()
    if not ret:
        print('ERROR: Cannot read first frame')
        sys.exit(1)

    _clone = _frame.copy()
    win = 'Click 4 road corners: TL -> TR -> BR -> BL'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_cb)

    print('\n=== Homography Calibration ===')
    print('Click 4 points on the road that form a real-world rectangle.')
    print('Order: top-left → top-right → bottom-right → bottom-left')
    print("Press 'r' to reset, 'q'/ESC to quit.\n")

    while True:
        cv2.imshow(win, _clone)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break

        if key == ord('r'):
            _points = []
            _clone = _frame.copy()
            print('Points reset.')

        if len(_points) == 4:
            src = np.array(_points, dtype=np.float32)
            dst = np.array([
                [0, 0],
                [BEV_W - 1, 0],
                [BEV_W - 1, BEV_H - 1],
                [0, BEV_H - 1],
            ], dtype=np.float32)

            H, _ = cv2.findHomography(src, dst)

            bev = cv2.warpPerspective(_frame, H, (BEV_W, BEV_H))
            cv2.imshow('BEV Preview', bev)

            flat = H.flatten().tolist()

            print(f'\n✓ Homography computed!  (BEV size: {BEV_W}×{BEV_H})\n')
            print('Source points (pixels):', _points)
            print()
            print('── Copy-paste for launch / CLI ──────────────────')
            print()
            # Python list format
            print(f'homography: {[round(v, 8) for v in flat]}')
            print(f'bev_width:  {BEV_W}')
            print(f'bev_height: {BEV_H}')
            print()
            # ROS2 CLI format
            vals = ','.join(f'{v:.8f}' for v in flat)
            print(f'-p homography:="[{vals}]" -p bev_width:={BEV_W} -p bev_height:={BEV_H}')
            print()
            print('── As YAML (for param file) ─────────────────────')
            print('  homography:')
            for v in flat:
                print(f'    - {v:.8f}')
            print()
            print("Press 'r' to redo, 'q'/ESC to quit.")

            # Wait for user to press a key before allowing redo
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('r'):
                    _points = []
                    _clone = _frame.copy()
                    cv2.destroyWindow('BEV Preview')
                    print('\nPoints reset – click 4 new points.\n')
                    break
                if k2 in (ord('q'), 27):
                    cv2.destroyAllWindows()
                    cap.release()
                    return

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
