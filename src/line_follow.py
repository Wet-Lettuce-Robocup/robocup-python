import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from components.cameras.down_camera import DownCamera


class Task(Enum):
    INIT = 0
    FOLLOW = 1
    TOWER = 2
    RESCUE = 3


@dataclass
class LineFollowResult:
    # Desired direction relative to the robot
    target_angle: float = 0.0

    # What the main state machine should currently do: FOLLOW / FORWARD / TURN_LEFT / TURN_RIGHT / U_TURN / REVERSE
    action: str = "FOLLOW"

    # Line information
    line_detected: bool = False
    line_angle: float = 0.0
    line_offset: float = 0.0

    # Gap information
    gap_detected: bool = False

    # Green marker information
    green_left: bool = False
    green_right: bool = False
    u_turn: bool = False

    # Junction / crossing information
    horizontal_line: bool = False
    vertical_line: bool = False
    horizontal_crossing: bool = False

    # Recovery information
    bottom_only: bool = False
    sharp_bend: bool = False
    stuck: bool = False
    recovering: bool = False

    # Counters
    no_line_frames: int = 0
    gap_frames: int = 0
    same_frame_frames: int = 0
    bottom_only_frames: int = 0

    # Last reliable line information
    last_line_angle: float = 0.0
    last_line_offset: float = 0.0

    # Optional debugging image.
    debug_frame: np.ndarray | None = field(default=None, repr=False)


class Follow:
    DEBUG = True

    # Camera
    WIDTH = 200
    HEIGHT = 100

    # Black line det
    BLUR_SIZE = 9

    MORPH_CLOSE_SIZE = 9
    MORPH_OPEN_SIZE = 3

    BLACK_THRESH = 60

    # Horizontal line det
    CROSSING_SIDE_Y_TOLERANCE = 8
    CROSSING_MIN_EDGE_RUN = 3
    CROSSING_MAX_EDGE_RUN = 25
    CROSSING_IGNORE_BAND = 8
    CROSSING_TOP_X_TOLERANCE = 45

    # Green turn det

    HOUGH_THRESHOLD = 15
    HOUGH_MIN_LINE_LENGTH = 18
    HOUGH_MAX_LINE_GAP = 8

    GREEN_H_LOW = 35
    GREEN_H_HIGH = 90
    GREEN_S_LOW = 70
    GREEN_V_LOW = 40

    GREEN_MIN_AREA = 2000
    GREEN_MAX_AREA = 15000

    GREEN_PIXEL_THRESHOLD = 400
    GREEN_HSV_DOWNSAMPLE = 2

    # Driving
    VELOCITY = 280
    OFFSET_GAIN = 15.0
    MAX_TARGET_ANGLE = 90.0

    # Gap and recovery
    NO_LINE_LIMIT = 5

    GAP_LIMIT = 20

    STUCK_LIMIT = 30
    SAME_FRAME_THRESHOLD = 0.8

    LOWER_LINE_LIMIT = 0.75
    LOWER_LINE_REVERSE_THRESH = 0.85

    MIN_BOTTOM_LINE_POINTS = 8
    BOTTOM_LINE_MIN_HEIGHT = 4

    BOTTOM_SHARP_ANGLE = 22.0
    BOTTOM_SHARP_HISTORY_ANGLE = 18.0
    BOTTOM_REVERSE_FRAMES = 12

    # Rescue detection
    MIN_RED = 200

    RED_S_MIN = 70
    RED_V_MIN = 70

    # PID
    MAX_TURN = 500

    KP = 5.5
    KI = 0.0
    KD = 0.2

    def __init__(self, i2c_controller, robot):
        self.logger = logging.getLogger("robot.line_follow")

        self.i2c_controller = i2c_controller
        self.robot = robot
        self.camera = DownCamera()

        self.raw_frame = None
        self.cropped_frame = None

        self.startTime = time.monotonic()

        self.follow_status = Task.INIT
        self.task_started = False

        # Last reliable line
        self.last_line_angle = 0.0
        self.last_line_offset = 0.0
        self.last_line_frame = None

        # State
        self.in_gap = False
        self.recovering = False

        # Bottom-only recovery state
        self.bottom_recovering = False
        self.bottom_only_frames = 0

        # Frame comparison
        self.previous_frame = None

        # Green
        self.last_green_centres = []

        # Counters
        self.no_line_frames = 0
        self.gap_frames = 0
        self.same_frame_frames = 0

        self.pid = PID(
            kp=self.KP,
            kd=self.KD,
            ki=self.KI,
        )

        self.last_time = None
        self.last_error = 0
        self.last_target_angle = 0.0

        self.debug_frame = None

        # Morphology
        self.black_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.MORPH_CLOSE_SIZE, self.MORPH_CLOSE_SIZE),
        )

        self.black_open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.MORPH_OPEN_SIZE, self.MORPH_OPEN_SIZE),
        )

        self.green_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        self.red_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

    def _transition_to(self, task):
        self.logger.info(f"Task: {self.follow_status.name} -> {task.name}")

        self.follow_status = task
        self.task_started = False

    def get_debug_frame(self):
        return self.debug_frame

    def process_line(self, raw_frame, cropped_frame, debug=False) -> LineFollowResult:
        if cropped_frame is None:
            return LineFollowResult(
                action="REVERSE",
                target_angle=self.last_target_angle,
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
        _, black_mask = self._make_black_mask(frame)

        crossing = self._detect_horizontal_crossing(black_mask)

        line_info = self._detect_line(black_mask, crossing)

        # Green processing
        green_present = self._green_present(frame)

        if green_present:
            green_mask = self._get_green_mask(
                frame,
                downsample=False,
            )

            geometry = self._detect_junction_geometry(black_mask)

            green_info = self._detect_green_squares(
                frame,
                geometry["horizontal_y"],
                geometry["vertical_x"],
                green_mask=green_mask,
            )

            special_result = self._handle_green_markers(
                green_info,
                geometry,
            )
        else:
            special_result = None
            geometry = {
                "horizontal": [],
                "vertical": [],
                "horizontal_y": None,
                "vertical_x": None,
            }
            green_info = {
                "centres": [],
                "green_left": False,
                "green_right": False,
                "mask": None,
            }

        if special_result is not None:
            result = special_result

            result.horizontal_crossing = crossing["detected"]

            result.debug_frame = (
                self._make_debug_frame(
                    frame,
                    black_mask,
                    line_info,
                    crossing,
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
                target_angle=self.last_target_angle,
                action="REVERSE",
                line_detected=line_info["detected"],
                line_angle=line_info["angle"],
                line_offset=line_info["offset"],
                horizontal_crossing=crossing["detected"],
                bottom_only=line_info["bottom_only"],
                stuck=True,
                recovering=True,
                no_line_frames=self.no_line_frames,
                gap_frames=self.gap_frames,
                same_frame_frames=same_frame,
                bottom_only_frames=self.bottom_only_frames,
                last_line_angle=self.last_line_angle,
                last_line_offset=self.last_line_offset,
            )

            result.debug_frame = (
                self._make_debug_frame(
                    frame,
                    black_mask,
                    line_info,
                    crossing,
                    geometry,
                    green_info,
                    result,
                )
                if debug
                else None
            )

            return result

        # Line detected
        if line_info["detected"]:
            self.no_line_frames = 0

            # Line found after a gap
            if self.in_gap:
                self.in_gap = False
                self.gap_frames = 0
                self.recovering = False

            # Bottom-only detection

            # There are two cases:
            #   1. Straight line / genuine gap: -> keep driving straight
            #   2. Sharp bend: -> command a hard turn
            # If the hard-turn state persists, assume the robot has
            # overshot and reverse.

            if line_info["bottom_only"]:
                self.bottom_only_frames += 1
                self.bottom_recovering = True

                current_angle = abs(line_info["angle"])

                previous_angle = abs(self.last_target_angle)

                sharp_bend = (
                    current_angle >= self.BOTTOM_SHARP_ANGLE
                    or previous_angle >= self.BOTTOM_SHARP_HISTORY_ANGLE
                )

                # Straight bottom-only line
                if not sharp_bend:
                    self.recovering = False

                    result = LineFollowResult(
                        target_angle=0.0,
                        action="FORWARD",
                        line_detected=True,
                        line_angle=line_info["angle"],
                        line_offset=line_info["offset"],
                        gap_detected=True,
                        horizontal_crossing=crossing["detected"],
                        bottom_only=True,
                        sharp_bend=False,
                        recovering=False,
                        no_line_frames=0,
                        gap_frames=self.gap_frames,
                        same_frame_frames=same_frame,
                        bottom_only_frames=self.bottom_only_frames,
                        last_line_angle=self.last_line_angle,
                        last_line_offset=self.last_line_offset,
                    )

                    if debug:
                        result.debug_frame = self._make_debug_frame(
                            frame,
                            black_mask,
                            line_info,
                            crossing,
                            geometry,
                            green_info,
                            result,
                        )

                    return result

                # Sharp bend
                if abs(line_info["angle"]) >= self.BOTTOM_SHARP_ANGLE:
                    turn_direction = np.sign(line_info["angle"])
                elif abs(self.last_target_angle) > 0:
                    turn_direction = np.sign(self.last_target_angle)
                else:
                    turn_direction = 1.0

                target_angle = float(90.0 * turn_direction)

                # Overshoot handling

                if self.bottom_only_frames >= self.BOTTOM_REVERSE_FRAMES:
                    self.recovering = True

                    result = LineFollowResult(
                        target_angle=target_angle,
                        action="REVERSE",
                        line_detected=True,
                        line_angle=line_info["angle"],
                        line_offset=line_info["offset"],
                        horizontal_crossing=crossing["detected"],
                        bottom_only=True,
                        sharp_bend=True,
                        recovering=True,
                        no_line_frames=0,
                        gap_frames=0,
                        same_frame_frames=same_frame,
                        bottom_only_frames=self.bottom_only_frames,
                        last_line_angle=self.last_line_angle,
                        last_line_offset=self.last_line_offset,
                    )

                    if debug:
                        result.debug_frame = self._make_debug_frame(
                            frame,
                            black_mask,
                            line_info,
                            crossing,
                            geometry,
                            green_info,
                            result,
                        )

                    return result

                # Hard turn in place
                self.recovering = True

                result = LineFollowResult(
                    target_angle=target_angle,
                    action="FOLLOW",
                    line_detected=True,
                    line_angle=line_info["angle"],
                    line_offset=line_info["offset"],
                    horizontal_crossing=crossing["detected"],
                    bottom_only=True,
                    sharp_bend=True,
                    recovering=True,
                    no_line_frames=0,
                    gap_frames=0,
                    same_frame_frames=same_frame,
                    bottom_only_frames=self.bottom_only_frames,
                    last_line_angle=self.last_line_angle,
                    last_line_offset=self.last_line_offset,
                )

                if debug:
                    result.debug_frame = self._make_debug_frame(
                        frame,
                        black_mask,
                        line_info,
                        crossing,
                        geometry,
                        green_info,
                        result,
                    )

                return result

            # A normal line
            self.bottom_only_frames = 0
            self.bottom_recovering = False
            self.recovering = False

            self._remember_line(line_info)

            target_angle = self._calculate_target_angle(line_info)

            result = LineFollowResult(
                target_angle=target_angle,
                action="FOLLOW",
                line_detected=True,
                line_angle=line_info["angle"],
                line_offset=line_info["offset"],
                horizontal_crossing=crossing["detected"],
                bottom_only=False,
                sharp_bend=False,
                no_line_frames=0,
                gap_frames=0,
                same_frame_frames=same_frame,
                bottom_only_frames=0,
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
                        target_angle=0.0,
                        action="FORWARD",
                        line_detected=False,
                        gap_detected=True,
                        horizontal_crossing=crossing["detected"],
                        recovering=False,
                        no_line_frames=self.no_line_frames,
                        gap_frames=self.gap_frames,
                        same_frame_frames=same_frame,
                        bottom_only_frames=0,
                        last_line_angle=self.last_line_angle,
                        last_line_offset=self.last_line_offset,
                    )

                else:
                    self.recovering = True

                    result = LineFollowResult(
                        target_angle=0.0,
                        action="REVERSE",
                        line_detected=False,
                        gap_detected=True,
                        horizontal_crossing=crossing["detected"],
                        recovering=True,
                        no_line_frames=self.no_line_frames,
                        gap_frames=self.gap_frames,
                        same_frame_frames=same_frame,
                        bottom_only_frames=0,
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
                        horizontal_crossing=crossing["detected"],
                        recovering=True,
                        no_line_frames=self.no_line_frames,
                        gap_frames=0,
                        same_frame_frames=same_frame,
                        bottom_only_frames=0,
                    )

                else:
                    result = LineFollowResult(
                        target_angle=0.0,
                        action="FORWARD",
                        line_detected=False,
                        horizontal_crossing=crossing["detected"],
                        no_line_frames=self.no_line_frames,
                        gap_frames=0,
                        same_frame_frames=same_frame,
                        bottom_only_frames=0,
                    )

        # Debug
        if debug:
            result.debug_frame = self._make_debug_frame(
                frame,
                black_mask,
                line_info,
                crossing,
                geometry,
                green_info,
                result,
            )

        return result

    # Black mask

    def _make_black_mask(self, frame):
        """
        Create a mask of dark regions.

        Red regions are removed so silver isn't included maybe.
        """

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        gray = cv2.GaussianBlur(
            gray,
            (self.BLUR_SIZE, self.BLUR_SIZE),
            0,
        )

        _, black_mask = cv2.threshold(
            gray,
            self.BLACK_THRESH,
            255,
            cv2.THRESH_BINARY_INV,
        )

        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_CLOSE,
            self.black_close_kernel,
        )

        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_OPEN,
            self.black_open_kernel,
        )

        # Remove red pixels
        red_mask = self._get_red_mask(frame)

        if red_mask is not None:
            black_mask[red_mask > 0] = 0

        return gray, black_mask

    # Red detection

    def _get_red_mask(self, image):
        if image is None:
            return None

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        mask1 = cv2.inRange(
            hsv,
            np.array(
                [0, self.RED_S_MIN, self.RED_V_MIN],
                dtype=np.uint8,
            ),
            np.array(
                [10, 255, 255],
                dtype=np.uint8,
            ),
        )

        mask2 = cv2.inRange(
            hsv,
            np.array(
                [170, self.RED_S_MIN, self.RED_V_MIN],
                dtype=np.uint8,
            ),
            np.array(
                [180, 255, 255],
                dtype=np.uint8,
            ),
        )

        red_mask = mask1 | mask2

        # Close small gaps in reflected red areas
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            self.red_kernel,
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            self.red_kernel,
        )

        return red_mask

    def red_detected(self, image):
        if image is None:
            return False

        red_mask = self._get_red_mask(image)

        if red_mask is None:
            return False

        red_pixels = cv2.countNonZero(red_mask)

        # Check num of red pixels
        if red_pixels >= self.MIN_RED:
            self.logger.info(f"Red pixels: {red_pixels}")
            return True

        # Then check for red contours
        contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= self.MIN_RED:
                self.logger.info(f"Red area: {area:.1f}")
                return True

        return False

    # Horizontal line detection

    def _find_edge_run(self, black_mask, edge):
        """
        Find the longest compact black run touching one image edge.

        Returns:
            {
                "detected": bool,
                "start": int,
                "end": int,
                "centre": float,
                "length": int,
            }
        """

        h, w = black_mask.shape

        if edge == "left":
            values = black_mask[:, 0] > 0

        elif edge == "right":
            values = black_mask[:, w - 1] > 0

        elif edge == "top":
            values = black_mask[0, :] > 0

        else:
            raise ValueError(f"Unknown edge: {edge}")

        indices = np.where(values)[0]

        if len(indices) == 0:
            return {
                "detected": False,
                "start": None,
                "end": None,
                "centre": None,
                "length": 0,
            }

        # Split indices into contiguous runs
        runs = []

        start = indices[0]
        previous = indices[0]

        for value in indices[1:]:
            if value != previous + 1:
                runs.append((start, previous))
                start = value

            previous = value

        runs.append((start, previous))

        # Keep plausible compact runs
        valid_runs = []

        for start, end in runs:
            length = end - start + 1

            if length < self.CROSSING_MIN_EDGE_RUN:
                continue

            if length > self.CROSSING_MAX_EDGE_RUN:
                continue

            valid_runs.append((length, start, end))

        if not valid_runs:
            return {
                "detected": False,
                "start": None,
                "end": None,
                "centre": None,
                "length": 0,
            }

        # Longest plausible run
        length, start, end = max(
            valid_runs,
            key=lambda x: x[0],
        )

        return {
            "detected": True,
            "start": start,
            "end": end,
            "centre": (start + end) / 2.0,
            "length": length,
        }

    def _detect_horizontal_crossing(self, black_mask):
        """
        Detect the type of junction described for the course:

            main line
                 |
                 |
        ----------+----------
                 |
                 |

        Looks for:
            - black exiting left edge
            - black exiting right edge
            - both exits at approximately the same Y
            - black exiting the top edge
            - top exit reasonably close to the centre
        """

        left = self._find_edge_run(
            black_mask,
            "left",
        )

        right = self._find_edge_run(
            black_mask,
            "right",
        )

        top = self._find_edge_run(
            black_mask,
            "top",
        )

        result = {
            "detected": False,
            "left_y": left["centre"],
            "right_y": right["centre"],
            "top_x": top["centre"],
            "crossing_y": None,
            "left": left,
            "right": right,
            "top": top,
        }

        if not left["detected"]:
            return result

        if not right["detected"]:
            return result

        if not top["detected"]:
            return result

        side_y_difference = abs(left["centre"] - right["centre"])

        if side_y_difference > self.CROSSING_SIDE_Y_TOLERANCE:
            return result

        centre_x = self.WIDTH / 2.0

        if abs(top["centre"] - centre_x) > self.CROSSING_TOP_X_TOLERANCE:
            return result

        result["detected"] = True

        result["crossing_y"] = (left["centre"] + right["centre"]) / 2.0

        return result

    # Line detection

    def _detect_line(
        self,
        black_mask,
        crossing=None,
    ):
        """
        Detect the main line to follow.

        Normal case:
            PCA over the main contour.

        Horizontal-crossing case:
            Ignore a band around the detected horizontal line before
            calculating PCA. This prevents the horizontal component
            from dominating the direction calculation.
        """

        h, w = black_mask.shape

        if crossing is None:
            crossing = {
                "detected": False,
                "crossing_y": None,
                "top_x": None,
            }

        # Ignore the top ~20% when choosing the line contour.
        roi_start = int(h * 0.2)

        roi = black_mask[roi_start:h, :]

        contours, _ = cv2.findContours(
            roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return self._empty_line_info(crossing)

        centre_x = w / 2.0
        bottom_y = roi.shape[0] - 1

        best_contour = None
        best_score = float("inf")

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 10:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)

            # Ignore very small regions
            if cw < 3 or ch < 3:
                continue

            # Check if contour is close to bottom centre of frame
            contour_bottom = y + ch
            contour_centre_x = x + cw / 2.0

            centre_distance = abs(contour_centre_x - centre_x)
            bottom_distance = abs(bottom_y - contour_bottom)

            # Normal contour scoring
            score = centre_distance * 3.0 + bottom_distance * 1.0 - area * 0.01

            # Prefer vertical lines
            if crossing["detected"]:
                top_x = crossing["top_x"]

                if top_x is not None and x <= top_x <= x + cw:
                    score -= 250.0

                score -= ch * 1.5

            if score < best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            return self._empty_line_info(crossing)

        points = best_contour.reshape(-1, 2).astype(np.float32)

        if len(points) < 12:
            return self._empty_line_info(crossing)

        # Convert back to full-frame coordinates
        points[:, 1] += roi_start

        # Check if line is fully within lower part of frame

        bottom_threshold = h * self.LOWER_LINE_REVERSE_THRESH

        min_y = float(np.min(points[:, 1]))
        max_y = float(np.max(points[:, 1]))

        line_height = max_y - min_y

        bottom_only = (
            len(points) >= self.MIN_BOTTOM_LINE_POINTS
            and min_y >= bottom_threshold
            and line_height >= self.BOTTOM_LINE_MIN_HEIGHT
        )

        # Calculate PCA points

        direction_points = points

        if crossing["detected"]:
            crossing_y = crossing["crossing_y"]

            if crossing_y is not None:
                keep = np.abs(points[:, 1] - crossing_y) > self.CROSSING_IGNORE_BAND

                filtered = points[keep]

                if len(filtered) >= 12:
                    direction_points = filtered

        mean, eigenvectors = cv2.PCACompute(
            direction_points,
            mean=None,
        )

        direction = eigenvectors[0]

        dx = float(direction[0])
        dy = float(direction[1])

        # Make direction point towards the bottom of the image
        if dy < 0:
            dx *= -1
            dy *= -1

        angle = -np.degrees(np.arctan2(dx, dy))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        bottom_start = int(h * self.LOWER_LINE_LIMIT)

        bottom_points = points[points[:, 1] >= bottom_start]

        if crossing["detected"]:
            crossing_y = crossing["crossing_y"]

            if crossing_y is not None:
                bottom_points = bottom_points[
                    np.abs(bottom_points[:, 1] - crossing_y) > self.CROSSING_IGNORE_BAND
                ]

        if len(bottom_points) >= 4:
            bottom_center = float(np.mean(bottom_points[:, 0]))

            offset = (bottom_center - (w / 2.0)) / (w / 2.0)

            offset = float(np.clip(offset, -1.0, 1.0))

        else:
            offset = 0.0

        return {
            "detected": True,
            "angle": float(angle),
            "offset": offset,
            "points": points,
            "direction_points": direction_points,
            "bottom_only": bottom_only,
            "crossing": crossing["detected"],
            "crossing_y": crossing["crossing_y"],
        }

    def _empty_line_info(self, crossing=None):

        if crossing is None:
            crossing = {
                "detected": False,
                "crossing_y": None,
            }

        return {
            "detected": False,
            "angle": 0.0,
            "offset": 0.0,
            "points": None,
            "direction_points": None,
            "bottom_only": False,
            "crossing": crossing["detected"],
            "crossing_y": crossing["crossing_y"],
        }

    # Target angle calculation

    def _calculate_target_angle(
        self,
        line_info,
    ):
        """
        Combine line direction and lateral position.
        """

        angle = line_info["angle"]
        offset = line_info["offset"]

        target = angle + offset * self.OFFSET_GAIN

        return float(
            np.clip(
                target,
                -self.MAX_TARGET_ANGLE,
                self.MAX_TARGET_ANGLE,
            )
        )

    def _remember_line(self, line_info):
        if line_info["bottom_only"]:
            return

        self.last_line_angle = line_info["angle"]
        self.last_line_offset = line_info["offset"]
        self.last_target_angle = self._calculate_target_angle(line_info)

        self.last_line_frame = True

    # Green handling

    def _get_green_mask(self, frame, downsample=False):

        if frame is None:
            return None

        if downsample:
            frame = frame[
                :: self.GREEN_HSV_DOWNSAMPLE,
                :: self.GREEN_HSV_DOWNSAMPLE,
            ]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

        return cv2.inRange(
            hsv,
            lower_green,
            upper_green,
        )

    def _green_present(self, frame):
        """Cheap green test before running the more expensive contour / Hough processing."""

        if frame is None:
            return False

        green_mask = self._get_green_mask(
            frame,
            downsample=True,
        )

        if green_mask is None:
            return False

        return cv2.countNonZero(green_mask) >= self.GREEN_PIXEL_THRESHOLD

    # Green turn geometry

    def _detect_junction_geometry(self, black_mask):
        """
        Use Hough lines for the green-marker geometry.
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
            for line in lines:
                x1, y1, x2, y2 = map(int, np.asarray(line).reshape(-1))

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

        # find the longest lines
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

    def _detect_green_squares(self, frame, horizontal_y, vertical_x, green_mask=None):
        """Detect green squares that are below a black line (valid green turns)."""
        if green_mask is None:
            green_mask = self._get_green_mask(frame)

        green_mask = cv2.morphologyEx(
            green_mask,
            cv2.MORPH_OPEN,
            self.green_kernel,
        )

        contours, _ = cv2.findContours(
            green_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        centres = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.GREEN_MIN_AREA or area > self.GREEN_MAX_AREA:
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
            tolerance = 5

            for cx, cy, area in centres:
                if cy <= horizontal_y:
                    continue

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

        # Both sides = U-turn
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

    # Stuck handling

    def _update_same_frame_counter(self, frame):
        """
        Check whether camera frames are essentially identical.
        """

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

    # Debug camera frame

    def _make_debug_frame(
        self, frame, black_mask, line_info, crossing, geometry, green_info, result
    ):
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

        if crossing["detected"]:
            crossing_y = crossing["crossing_y"]

            top_x = crossing["top_x"]

            if crossing_y is not None:
                y = int(crossing_y)

                cv2.line(
                    debug,
                    (0, y),
                    (self.WIDTH - 1, y),
                    (0, 255, 255),
                    2,
                )

            if top_x is not None:
                x = int(top_x)

                cv2.line(
                    debug,
                    (x, 0),
                    (x, self.HEIGHT - 1),
                    (0, 255, 255),
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

        if line_info.get("direction_points") is not None:
            for x, y in line_info["direction_points"][::4]:
                cv2.circle(
                    debug,
                    (int(x), int(y)),
                    2,
                    (255, 0, 0),
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
            f"line: {result.line_angle:.1f}",
            (3, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            debug,
            f"offset: {result.line_offset:.2f}",
            (3, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            debug,
            f"gap: {result.gap_frames}",
            (3, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            debug,
            f"bottom: {result.bottom_only_frames}",
            (3, 77),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

        status = []

        if result.horizontal_crossing:
            status.append("CROSS")

        if result.bottom_only:
            status.append("BOTTOM")

        if result.sharp_bend:
            status.append("BEND")

        if result.gap_detected:
            status.append("GAP")

        if result.recovering:
            status.append("RECOVER")

        cv2.putText(
            debug,
            " ".join(status),
            (3, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

        return debug

    # Line in frame for water tower

    def lineInFrame(self) -> bool:

        frame = self.cropped_frame

        if frame is None:
            return False

        line = cv2.inRange(
            frame,
            (0, 0, 0),
            (45, 45, 45),
        )

        return cv2.countNonZero(line) > 5000

    # Main loop

    def main(self):
        now = time.monotonic()
        if self.last_time is None:
            dt = 0.0
        else:
            dt = now - self.last_time

        self.last_time = now

        self.raw_frame, self.cropped_frame = self.camera.get_frame()
        r_frame = self.raw_frame
        c_frame = self.cropped_frame

        if self.robot.limit_switch_pressed() and self.follow_status != Task.TOWER:
            self.logger.info("Limit switch pressed")
            self.robot.stop_moving()
            self._transition_to(Task.TOWER)

        elif self.follow_status == Task.TOWER:
            if not self.task_started:
                self.task_started = True

                self.robot.drive_dist_enc(-50, 300)
                time.sleep(1.5)

                self.robot.spin_enc(90, 550)
                time.sleep(3)
                self.robot.drive_PID(600, -490)
                time.sleep(5)
                self.robot.drive_PID(200)

            if self.lineInFrame():
                self.robot.stop_moving()
                self._transition_to(Task.FOLLOW)

        elif self.follow_status == Task.FOLLOW:
            if not self.task_started:
                self.pid.reset()
                self.last_time = None
                self.last_error = 0
                self.recovering = False
                self.bottom_recovering = False
                self.bottom_only_frames = 0
                self.in_gap = False
                self.no_line_frames = 0
                self.last_line_frame = None
                self.gap_frames = 0
                self.same_frame_frames = 0

                self.task_started = True

            # Rescue detection before line processing
            if self.red_detected(c_frame):
                self.logger.info("Red detected")
                self._transition_to(Task.RESCUE)
                return

            result = self.process_line(r_frame, c_frame, debug=self.DEBUG)

            if self.DEBUG and result.debug_frame is not None:
                self.debug_frame = result.debug_frame

            if result.action == "FOLLOW":
                error_pid = self.pid.update(result.target_angle, dt)

                turn_error = np.clip(
                    error_pid,
                    -self.MAX_TURN,
                    self.MAX_TURN,
                )

                turn_strength = abs(turn_error) / self.MAX_TURN

                # Makes robot slow on large error values
                turn_factor = 1.6  # Larger value = slows down more at larger angles
                speed_scale = 1.0 / (1.0 + turn_factor * turn_strength**2)
                velocity = self.VELOCITY * speed_scale

                # At very large steering errors, turn on the spot
                if turn_strength > 0.9:
                    velocity = 0

                self.robot.drive_PID(int(velocity), int(turn_error))

                self.last_error = turn_error
                # if self.DEBUG:
                #     self.logger.info(
                #         f"PID Error: {turn_error}, Target Angle: {result.target_angle}, Line Angle: {result.line_angle}, Line Offset: {result.line_offset}"
                #     )

            elif result.action == "FORWARD":
                self.logger.info("Moving forward")
                self.robot.drive_PID(int(self.VELOCITY * 0.5), 0)

            elif result.action == "TURN_LEFT" or result.action == "TURN_RIGHT":
                self.logger.info(f"Green turn detected {result.action}")

                self.robot.drive_dist_enc(60, 600)
                time.sleep(0.8)
                self.robot.spin_enc(result.target_angle - 20, 500)
                time.sleep(2.5)
                self.robot.drive_dist_enc(20, 400)

                self.pid.reset()
                self.last_time = None

            elif result.action == "U_TURN":
                self.logger.info("U-turn detected")

                self.robot.spin_enc(result.target_angle)
                time.sleep(5)
                # self.robot.drive_dist_enc(50)

                self.pid.reset()
                self.last_time = None

            elif result.action == "REVERSE":
                self.logger.info("Reversing")
                self.robot.drive_PID(-200)

        elif self.follow_status == Task.INIT:
            if not self.task_started:
                self._transition_to(Task.FOLLOW)
                self.task_started = True

    def is_finished(self):
        return self.follow_status == Task.RESCUE


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.previous_error = 0.0

        self.initialised = False

    def update(self, error, dt):
        if dt <= 0:
            return 0.0

        dt = min(dt, 0.1)

        if not self.initialised:
            self.previous_error = error
            self.initialised = True
            return self.kp * error

        self.integral += error * dt
        self.integral = np.clip(
            self.integral,
            -1000.0,
            1000.0,
        )

        derivative = (error - self.previous_error) / dt

        self.previous_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.initialised = False
