import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from components.cameras.down_camera import DownCamera


class Task(Enum):
    INIT = 1
    FOLLOW = 2
    TOWER = 3
    RESCUE = 4


@dataclass
class LineFollowResult:
    # Desired direction relative to the robot.
    target_angle: float = 0.0

    # What the main state machine should currently do: FOLLOW / FORWARD / TURN_LEFT / TURN_RIGHT / U_TURN / REVERSE
    action: str = "FOLLOW"

    # Useful information for debugging / state-machine decisions.
    line_detected: bool = False
    line_angle: float = 0.0
    line_offset: float = 0.0

    gap_detected: bool = False
    green_left: bool = False
    green_right: bool = False
    u_turn: bool = False

    horizontal_line: bool = False
    vertical_line: bool = False

    stuck: bool = False
    recovering: bool = False

    no_line_frames: int = 0
    gap_frames: int = 0
    same_frame_frames: int = 0

    # Last reliable line information.
    last_line_angle: float = 0.0
    last_line_offset: float = 0.0

    # Optional debugging image.
    debug_frame: np.ndarray | None = field(default=None, repr=False)


class Follow:
    VELOCITY = 50
    MIN_RED_AREA = 500.0

    WIDTH = 240
    HEIGHT = 135

    BLUR_SIZE = 5
    MORPH_CLOSE_SIZE = 5
    MORPH_OPEN_SIZE = 3
    HOUGH_THRESHOLD = 15
    HOUGH_MIN_LINE_LENGTH = 18
    HOUGH_MAX_LINE_GAP = 8

    BLACK_THRESH = 85
    GREEN_H_LOW = 35
    GREEN_H_HIGH = 90
    GREEN_S_LOW = 70
    GREEN_V_LOW = 40

    GREEN_MIN_AREA = 8
    GREEN_MAX_AREA = 1000

    OFFSET_GAIN = 40.0

    MAX_TARGET_ANGLE = 90.0

    NO_LINE_LIMIT = 3
    GAP_LIMIT = 10
    STUCK_LIMIT = 20
    SAME_FRAME_THRESHOLD = 1.5

    def __init__(self, i2c_controller, robot):
        self.logger = logging.getLogger("line_follow")

        self.i2c_controller = i2c_controller
        self.robot = robot
        self.camera = DownCamera()

        self.raw_frame = None
        self.cropped_frame = None

        self.startTime = time.monotonic()

        self.follow_status = Task.INIT
        self.task_started = False

        self.last_line_angle = 0.0
        self.last_line_offset = 0.0
        self.last_line_frame = None

        self.in_gap = False
        self.recovering = False

        self.previous_frame = None
        self.last_green_centres = []

        # Counters
        self.no_line_frames = 0
        self.gap_frames = 0
        self.same_frame_frames = 0

    def _transition_to(self, task):
        self.logger.info(f"Task: {self.follow_statuss.name} -> {task.name}")

        self.follow_status = task
        self.task_started = False

    def process_line(self, raw_frame, cropped_frame, debug=False) -> LineFollowResult:
        if cropped_frame is None:
            return LineFollowResult(
                action="REVERSE",
                target_angle=self.last_line_angle,
                recovering=True,
            )

        frame = cropped_frame

        if frame.shape[1] != self.WIDTH or frame.shape[0] != self.HEIGHT:
            self.logger.error(f"Frame is not the correct size {frame.shape}")
            frame = cv2.resize(
                frame,
                (self.WIDTH, self.HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

        # Check if frame hasn't changed -> robot is stuck
        same_frame = self._update_same_frame_counter(frame)

        # Black processing
        gray, black_mask = self._make_black_mask(frame)
        line_info = self._detect_line(black_mask)

        geometry = self._detect_junction_geometry(black_mask)

        # Green processing
        green_info = self._detect_green_squares(
            frame,
            geometry["horizontal_y"],
            geometry["vertical_x"],
        )

        special_result = self._handle_green_markers(
            green_info,
            geometry,
        )

        if special_result is not None:
            result = special_result
            result.debug_frame = (
                self._make_debug_frame(
                    frame,
                    black_mask,
                    line_info,
                    geometry,
                    green_info,
                    result,
                )
                if debug
                else None
            )

            if line_info["detected"]:
                self._remember_line(line_info)
            return result

        # Stuck detection
        if same_frame >= self.STUCK_LIMIT:
            self.recovering = True

            result = LineFollowResult(
                target_angle=self.last_line_angle,
                action="REVERSE",
                line_detected=line_info["detected"],
                line_angle=line_info["angle"],
                line_offset=line_info["offset"],
                stuck=True,
                recovering=True,
                no_line_frames=self.no_line_frames,
                gap_frames=self.gap_frames,
                same_frame_frames=same_frame,
                last_line_angle=self.last_line_angle,
                last_line_offset=self.last_line_offset,
            )

            result.debug_frame = (
                self._make_debug_frame(
                    frame,
                    black_mask,
                    line_info,
                    geometry,
                    green_info,
                    result,
                )
                if debug
                else None
            )

            return result

        # Only a black line
        if line_info["detected"]:
            self.no_line_frames = 0
            self._remember_line(line_info)

            # Line found after a gap
            if self.in_gap:
                self.in_gap = False
                self.gap_frames = 0
                self.recovering = False

            target_angle = self._calculate_target_angle(line_info)

            result = LineFollowResult(
                target_angle=target_angle,
                action="FOLLOW",
                line_detected=True,
                line_angle=line_info["angle"],
                line_offset=line_info["offset"],
                no_line_frames=0,
                gap_frames=0,
                same_frame_frames=same_frame,
                last_line_angle=self.last_line_angle,
                last_line_offset=self.last_line_offset,
            )

        # No line detected
        else:
            self.no_line_frames += 1

            # Line disappeared -> probably a gap

            if self.last_line_frame is not None:
                self.in_gap = True
                self.gap_frames += 1

                if self.gap_frames <= self.GAP_LIMIT:
                    result = LineFollowResult(
                        target_angle=self.last_line_angle,
                        action="FORWARD",
                        line_detected=False,
                        gap_detected=True,
                        recovering=False,
                        no_line_frames=self.no_line_frames,
                        gap_frames=self.gap_frames,
                        same_frame_frames=same_frame,
                        last_line_angle=self.last_line_angle,
                        last_line_offset=self.last_line_offset,
                    )

                else:
                    # Gap too big -> reverse to previous line
                    self.recovering = True

                    result = LineFollowResult(
                        target_angle=self.last_line_angle,
                        action="REVERSE",
                        line_detected=False,
                        gap_detected=True,
                        recovering=True,
                        no_line_frames=self.no_line_frames,
                        gap_frames=self.gap_frames,
                        same_frame_frames=same_frame,
                        last_line_angle=self.last_line_angle,
                        last_line_offset=self.last_line_offset,
                    )

            # No previous line
            else:
                if self.no_line_frames >= self.NO_LINE_LIMIT:
                    self.recovering = True

                    result = LineFollowResult(
                        target_angle=0.0,
                        action="REVERSE",
                        line_detected=False,
                        recovering=True,
                        no_line_frames=self.no_line_frames,
                        gap_frames=0,
                        same_frame_frames=same_frame,
                    )

                else:
                    result = LineFollowResult(
                        target_angle=0.0,
                        action="FORWARD",
                        line_detected=False,
                        no_line_frames=self.no_line_frames,
                        gap_frames=0,
                        same_frame_frames=same_frame,
                    )

        if debug:
            result.debug_frame = self._make_debug_frame(
                frame,
                black_mask,
                line_info,
                geometry,
                green_info,
                result,
            )

        return result

    def _make_black_mask(self, frame):
        """
        Blur the frame and create a mask of dark regions.
        """

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Blur removes variations in brightness
        gray = cv2.GaussianBlur(
            gray,
            (self.BLUR_SIZE, self.BLUR_SIZE),
            0,
        )

        _, black_mask = cv2.threshold(
            gray,
            self.BLACK_THRESHOLD,
            255,
            cv2.THRESH_BINARY_INV,
        )

        # Connect small gaps in the line.
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self.MORPH_CLOSE_SIZE,
                self.MORPH_CLOSE_SIZE,
            ),
        )

        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        # Remove isolated regions
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self.MORPH_OPEN_SIZE,
                self.MORPH_OPEN_SIZE,
            ),
        )

        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_OPEN,
            open_kernel,
        )

        return gray, black_mask

    def _detect_line(self, black_mask):
        """
        Detect the main line using dark pixels in the lower part of
        the image.
        """

        h, w = black_mask.shape

        # Ignore the top part of the image. will probably change tbh
        roi_start = int(h * 0.2)

        roi = black_mask[roi_start:h, :]

        ys, xs = np.nonzero(roi)

        if len(xs) < 12:
            return {
                "detected": False,
                "angle": 0.0,
                "offset": 0.0,
                "points": None,
            }

        # Reject frames where almost the entire ROI is black.
        black_fraction = len(xs) / float(roi.size)

        if black_fraction > 0.55:
            return {
                "detected": False,
                "angle": 0.0,
                "offset": 0.0,
                "points": None,
            }

        # Convert coordinates back to full-frame coordinates.
        ys = ys + roi_start

        points = np.column_stack((xs, ys)).astype(np.float32)

        # PCA gives the dominant direction of the line.
        mean, eigenvectors = cv2.PCACompute(
            points,
            mean=None,
        )

        direction = eigenvectors[0]

        dx = float(direction[0])
        dy = float(direction[1])

        # direction towards the bottom of the image
        if dy < 0:
            dx *= -1
            dy *= -1

        angle = np.degrees(np.arctan2(dx, dy))

        # Find where the line is near the bottom of the image
        bottom_start = int(h * 0.78)

        bottom_mask = black_mask[bottom_start:h, :]

        bottom_y, bottom_x = np.nonzero(bottom_mask)

        if len(bottom_x) >= 4:
            bottom_center = float(np.mean(bottom_x))

            offset = (bottom_center - (w / 2.0)) / (w / 2.0)

            # Clamp.
            offset = float(np.clip(offset, -1.0, 1.0))

        else:
            offset = 0.0

        return {
            "detected": True,
            "angle": float(angle),
            "offset": offset,
            "points": points,
        }

    def _calculate_target_angle(self, line_info):
        """
        Combine line direction and lateral position.

        Example:
            line angle = +5 degrees
            line is 10% right of centre

        -> target a little more to the right.
        """

        angle = line_info["angle"]
        offset = line_info["offset"]

        target = angle + (offset * self.OFFSET_GAIN)

        return float(
            np.clip(
                target,
                -self.MAX_TARGET_ANGLE,
                self.MAX_TARGET_ANGLE,
            )
        )

    def _remember_line(self, line_info):
        self.last_line_angle = line_info["angle"]
        self.last_line_offset = line_info["offset"]

        self.last_line_frame = True

    def _detect_junction_geometry(self, black_mask):
        """
        Use a small Hough transform to find long horizontal and vertical
        black lines for special green-marker detection.
        """

        lines = cv2.HoughLinesP(
            black_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=self.HOUGH_THRESHOLD,
            minLineLength=self.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=self.HOUGH_MAX_LINE_GAP,
        )

        horizontal = []
        vertical = []

        if lines is not None:
            for line in lines[:, 0]:
                x1, y1, x2, y2 = map(int, line)

                dx = x2 - x1
                dy = y2 - y1

                length = np.hypot(dx, dy)

                if length < self.HOUGH_MIN_LINE_LENGTH:
                    continue

                # horizontal
                if abs(dy) < length * 0.25:
                    horizontal.append((
                        length,
                        x1,
                        y1,
                        x2,
                        y2,
                    ))

                # vertical
                elif abs(dx) < length * 0.25:
                    vertical.append((
                        length,
                        x1,
                        y1,
                        x2,
                        y2,
                    ))

        # longest lines
        horizontal.sort(reverse=True)
        vertical.sort(reverse=True)

        horizontal_y = None
        vertical_x = None

        if horizontal:
            _, x1, y1, x2, y2 = horizontal[0]

            horizontal_y = (y1 + y2) / 2.0

        if vertical:
            _, x1, y1, x2, y2 = vertical[0]

            vertical_x = (x1 + x2) / 2.0

        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "horizontal_y": horizontal_y,
            "vertical_x": vertical_x,
        }

    def _detect_green_squares(self, frame, horizontal_y, vertical_x):
        """
        Detect green squares that are below a black line (valid green turns).
        """

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        lower_green = np.array(
            [
                self.GREEN_H_LOW,
                self.GREEN_S_LOW,
                self.GREEN_V_LOW,
            ],
            dtype=np.uint8,
        )

        upper_green = np.array(
            [
                self.GREEN_H_HIGH,
                255,
                255,
            ],
            dtype=np.uint8,
        )

        green_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        green_mask = cv2.morphologyEx(
            green_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        contours, _ = cv2.findContours(
            green_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        centres = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.GREEN_MIN_AREA:
                continue

            if area > self.GREEN_MAX_AREA:
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(
                contour,
                0.04 * perimeter,
                True,
            )

            # It should have 4 corners if it's a square
            if not (4 <= len(approx) <= 6):
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if h == 0:
                continue

            aspect = w / float(h)

            # Filter out elongated shapes
            if aspect < 0.55 or aspect > 1.8:
                continue

            fill_ratio = area / float(w * h)

            if fill_ratio < 0.45:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0

            centres.append((cx, cy, area))

        green_left = False
        green_right = False

        if horizontal_y is not None and vertical_x is not None:
            for cx, cy, area in centres:
                if cy <= horizontal_y:
                    continue

                # Small tolerance around vertical line
                tolerance = 5

                if cx < vertical_x - tolerance:
                    green_left = True

                elif cx > vertical_x + tolerance:
                    green_right = True

        return {
            "centres": centres,
            "green_left": green_left,
            "green_right": green_right,
            "mask": green_mask,
        }

    def _handle_green_markers(self, green_info, geometry):
        left = green_info["green_left"]
        right = green_info["green_right"]

        horizontal_exists = geometry["horizontal_y"] is not None
        vertical_exists = geometry["vertical_x"] is not None

        if not horizontal_exists or not vertical_exists:
            return None

        # Both sides = uturn
        if left and right:
            return LineFollowResult(
                target_angle=180.0,
                action="U_TURN",
                line_detected=True,
                u_turn=True,
                green_left=True,
                green_right=True,
                horizontal_line=True,
                vertical_line=True,
                last_line_angle=self.last_line_angle,
                last_line_offset=self.last_line_offset,
            )

        if left:
            return LineFollowResult(
                target_angle=-90.0,
                action="TURN_LEFT",
                line_detected=True,
                green_left=True,
                green_right=False,
                horizontal_line=True,
                vertical_line=True,
                last_line_angle=self.last_line_angle,
                last_line_offset=self.last_line_offset,
            )

        if right:
            return LineFollowResult(
                target_angle=90.0,
                action="TURN_RIGHT",
                line_detected=True,
                green_left=False,
                green_right=True,
                horizontal_line=True,
                vertical_line=True,
                last_line_angle=self.last_line_angle,
                last_line_offset=self.last_line_offset,
            )

        return None

    def _update_same_frame_counter(self, frame):
        """Check if the camera frame has changed substantially or not."""

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        if self.previous_frame is None:
            self.previous_frame = gray
            self.same_frame_frames = 0
            return 0

        diff = cv2.absdiff(
            gray,
            self.previous_frame,
        )

        mean_difference = float(np.mean(diff))

        if mean_difference < self.SAME_FRAME_THRESHOLD:
            self.same_frame_frames += 1
        else:
            self.same_frame_frames = 0

        self.previous_frame = gray

        return self.same_frame_frames

    def _make_debug_frame(self, frame, black_mask, line_info, geometry, green_info, result):
        debug = frame.copy()

        for line in geometry["horizontal"]:
            _, x1, y1, x2, y2 = line

            cv2.line(
                debug,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                1,
            )

        for line in geometry["vertical"]:
            _, x1, y1, x2, y2 = line

            cv2.line(
                debug,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                1,
            )

        if geometry["horizontal_y"] is not None:
            y = int(geometry["horizontal_y"])

            cv2.line(
                debug,
                (0, y),
                (self.WIDTH - 1, y),
                (255, 255, 0),
                1,
            )

        if geometry["vertical_x"] is not None:
            x = int(geometry["vertical_x"])

            cv2.line(
                debug,
                (x, 0),
                (x, self.HEIGHT - 1),
                (255, 0, 255),
                1,
            )

        for cx, cy, area in green_info["centres"]:
            cv2.circle(
                debug,
                (int(cx), int(cy)),
                5,
                (0, 255, 0),
                2,
            )

        if line_info["points"] is not None:
            for x, y in line_info["points"][::4]:
                cv2.circle(
                    debug,
                    (int(x), int(y)),
                    1,
                    (0, 0, 255),
                    -1,
                )

        cv2.line(
            debug,
            (self.WIDTH // 2, 0),
            (self.WIDTH // 2, self.HEIGHT - 1),
            (255, 255, 255),
            1,
        )

        cv2.putText(
            debug,
            f"action: {result.action}",
            (3, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            debug,
            f"angle: {result.target_angle:.1f}",
            (3, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            debug,
            f"gap: {result.gap_frames}",
            (3, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            debug,
            f"same: {result.same_frame_frames}",
            (3, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

        return debug

    def lineInFrame(self) -> bool:
        frame = self.cropped_frame

        if frame is None:
            return False

        line = cv2.inRange(self.img, (0, 0, 0), (45, 45, 45))
        return cv2.countNonZero(line) > 5000

    def red_detected(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red has two ranges in HSV because hue wraps around at 180
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2

        # Create 5x5 rectangular kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Morphological opening
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find external contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_detected = False

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.MIN_RED_AREA:
                continue

            red_detected = True

        return red_detected

    def main(self):
        if self.robot.limit_switch_pressed():
            self.robot.stop_moving()
            self._transition_to(Task.TOWER)

        elif self.follow_status == Task.TOWER:
            if not self.task_started:
                self.task_started = True

                self.robot.spin(90)
                self.robot.drive_dist(30, -10, 40)  # to tune
                self.robot.drive(20)

            if self.lineInFrame():
                self.robot.stop_moving()
                self._transition_to(Task.FOLLOW)

        elif self.follow_status == Task.FOLLOW:
            self.raw_frame, self.cropped_frame = self.camera.get_frame()
            r_frame = self.raw_frame
            c_frame = self.cropped_frame

            if self.red_detected(c_frame):
                self._transition_to(Task.RESCUE)
                return

            result = self.process_line(r_frame, c_frame, debug=True)

            cv2.imshow("Debug", result.debug_frame)
            cv2.waitKey(1)

            if result.action == "FOLLOW":
                # angle = pid.calcTurnRate(angle, 1.4, 0, 0, self.lastError, self.pastErrors)
                self.robot.drive(self.VELOCITY, result.target_angle)
            elif result.action == "FORWARD":
                self.robot.drive(self.VELOCITY * (2 / 3), result.target_angle)
            elif result.action == "TURN_LEFT" or result.action == "TURN_RIGHT":
                self.robot.spin(result.target_angle)
            elif result.action == "U_TURN":
                self.robot.stop_moving()
                self.robot.spin(result.target_angle)
                self.robot.drive_dist(0.05)
            elif result.action == "REVERSE":
                self.robot.drive(-20)

        elif self.follow_status == Task.INIT:
            if not self.task_started:
                self._transition_to(Task.FOLLOW)
                self.task_started = True

    def is_finished(self):
        return self.follow_status == Task.RESCUE
