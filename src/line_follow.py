import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from components.cameras.down_camera import DownCamera


class Task(Enum):
    INIT = 0
    FOLLOW = 1
    TOWER = 2
    EXIT = 3


class LineState(Enum):
    NO_LINE = 0
    LINE = 1
    INTERSECTION = 2
    INTERSECTION_GREEN = 3
    GAP_START = 5
    GAP = 6
    GAP_WITH_LINE = 7
    GAP_END = 8
    U_TURN = 9  # green markers on both sides (original green U-turn detection)


@dataclass
class LineModeConfig:
    """Per-call switches for the line processor (same fields as center_processor)."""

    generate_debug_frame: bool = False
    calculate_green_center: bool = False
    gap_crop_enabled: bool = True
    ignore_last_start: bool = False
    ignore_last_end: bool = False
    ignore_intersections: bool = False
    ignore_intersections_with_green: bool = False
    # Reference point (relative 0..1) the steering angle is measured from.
    # (0.5, 1.0) = bottom centre of the image -> angle 0 means "straight ahead".
    angle_calculation_center_x: float = 0.5
    angle_calculation_center_y: float = 1.0


@dataclass
class LineFollowResult:
    state: LineState = LineState.NO_LINE
    # Steering angle in RADIANS, 0 = straight ahead, positive = line is to the RIGHT.
    angle: float = 0.0
    line_angle: float = 0.0
    start_angle: float = 0.0
    black_pixel_count: int = 0
    end_pixel_count: int = 0
    edge_contour_count: int = 0
    start_pixel_count: int = 0
    p_in: Optional[Tuple[float, float]] = None
    p_out: Optional[Tuple[float, float]] = None
    p_green: Optional[Tuple[float, float]] = None
    horizontal_crossing: bool = False  # original crossing detector (flag only)
    time_stamp: float = 0.0
    debug_frame: Optional[np.ndarray] = field(default=None, repr=False)


# ======================================================================
# Image helpers.
# NOTE: center_processor.py imported these from biobrause.hardware.camera.line.util,
# which was not provided, so they are reimplemented here from how they are used.
# If you have the original util module, swap these for the originals.
# ======================================================================


def adaptive_threshold(gray, block_size, c, invert=True):
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
        block_size,
        c,
    )


def morph_transform_image(binary, erode1, dilate, erode2, k_erode1, k_dilate, k_erode2):
    out = cv2.erode(binary, k_erode1, iterations=erode1)
    out = cv2.dilate(out, k_dilate, iterations=dilate)
    return cv2.erode(out, k_erode2, iterations=erode2)


def subtract_binaries(a, b):
    return cv2.subtract(a, b)


def find_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def filter_contours_by_area(contours, min_area):
    kept, removed = [], []
    for c in contours:
        (kept if cv2.contourArea(c) >= min_area else removed).append(c)
    return kept, removed


def _centroid(contour) -> Tuple[int, int]:
    m = cv2.moments(contour)
    if m["m00"] > 0:
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    x, y, w, h = cv2.boundingRect(contour)
    return x + w // 2, y + h // 2


def filter_contours_by_relative_area(contours, gray, x_min, x_max, y_min, y_max):
    h, w = gray.shape[:2]
    kept = []
    for c in contours:
        cx, cy = _centroid(c)
        if x_min * w <= cx <= x_max * w and y_min * h <= cy <= y_max * h:
            kept.append(c)
    return kept


def contour2binary(contour, gray):
    out = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.drawContours(out, [contour], -1, 255, thickness=cv2.FILLED)
    return out


def get_edge_binary(binary, thickness=2):
    """Parts of the shape that touch the image border (where the line enters/leaves)."""
    border = np.zeros_like(binary)
    border[:thickness, :] = 255
    border[-thickness:, :] = 255
    border[:, :thickness] = 255
    border[:, -thickness:] = 255
    return cv2.bitwise_and(binary, border)


class ContourHandler:
    def __init__(self, contour, gray):
        self.contour = contour
        self.gray = gray
        self.center = _centroid(contour)
        self.binary = contour2binary(contour, gray)
        self.area = float(cv2.countNonZero(self.binary))
        self.scaled_binary = self.binary

    def scale_binary(self, pixels):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
        self.scaled_binary = cv2.dilate(self.binary, k)


def contour_touches_border(gray, contour):
    h, w = gray.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    return x <= 0 or y <= 0 or x + bw >= w or y + bh >= h


def overlap_binaries(binaries):
    out = binaries[0].copy()
    for b in binaries[1:]:
        out = cv2.bitwise_and(out, b)
    return out


def check_overlap_binaries(binaries):
    return cv2.countNonZero(overlap_binaries(binaries)) > 0


def check_overlap_contours(handlers):
    return check_overlap_binaries([h.binary for h in handlers])


def calculate_distance_between_points(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def calculate_manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_closest_contour_handler(handlers, point, manhatten=True):
    if not handlers:
        return None
    dist = calculate_manhattan_distance if manhatten else calculate_distance_between_points
    return min(handlers, key=lambda h: dist(h.center, point))


def get_closest_contour_to_contour(contours, reference):
    ref = _centroid(reference)
    return min(contours, key=lambda c: calculate_distance_between_points(_centroid(c), ref))


def get_largest_contour(contours):
    return max(contours, key=cv2.contourArea)


def is_point_in_image(point, image):
    h, w = image.shape[:2]
    return 0 <= point[0] < w and 0 <= point[1] < h


def is_point_near_border(point, image, margin):
    h, w = image.shape[:2]
    return (
        point[0] < margin or point[0] >= w - margin or point[1] < margin or point[1] >= h - margin
    )


def get_angle_between_points(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def get_point_in_direction(point, angle, distance):
    return int(point[0] + distance * math.cos(angle)), int(point[1] + distance * math.sin(angle))


def find_border_intersection(image, p1, p2):
    """Extend the ray p1 -> p2 until it leaves the image and return that border point."""
    h, w = image.shape[:2]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return int(p2[0]), int(p2[1])
    ts = []
    if dx > 0:
        ts.append((w - 1 - p1[0]) / dx)
    elif dx < 0:
        ts.append((0 - p1[0]) / dx)
    if dy > 0:
        ts.append((h - 1 - p1[1]) / dy)
    elif dy < 0:
        ts.append((0 - p1[1]) / dy)
    t = min(ts)
    return int(p1[0] + dx * t), int(p1[1] + dy * t)


def clip_line(image, p1, p2):
    h, w = image.shape[:2]
    ok, a, b = cv2.clipLine((0, 0, w, h), tuple(map(int, p1)), tuple(map(int, p2)))
    return (a, b) if ok else (p1, p2)


class Follow:
    # ==================================================================
    #                       TUNABLE PARAMETERS
    # Everything you are likely to tune is in this block.
    # ==================================================================

    # ---- Camera ------------------------------------------------------
    # DownCamera captures 240x135 and crops [17:117, 20:220] -> 200x100. That is already small
    # (20k pixels), so frames are NOT downscaled further: halving would shrink the line to ~11 px
    # and force re-scaling every kernel/iteration count for almost no speed gain.
    WIDTH = 200  # expected frame width (frames are resized to this)
    HEIGHT = 100  # expected frame height

    # ---- [TUNE] Line size at 200x100 ----------------------------------
    # ESTIMATE from the previous code (green squares >= 3000 px^2 => ~55 px sides, crossing edge run
    # <= 25 px): the black line is ~22 px wide, green markers ~2.5x that. Enable debug and read the
    # "Measured line width" log, then set this to the measured value. The size-dependent values
    # below (AREA_MIN_*, EXPAND_*, GAP_START_LOCAL_MIN_PIXELS) follow it automatically.
    LINE_WIDTH_PX = 33  # measured from a real snapshot: 32-35 px wide, top bar 29-35 px thick

    # ---- [TUNE] Black line thresholding -------------------------------
    # "fixed"    = previous version's method: pixel is black if gray < BLACK_THRESH_FIXED (default, worked on this camera)
    # "adaptive" = center_processor's method: black if darker than local mean by ADAPTIVE_THRESHOLD_C
    BLACK_THRESHOLD_MODE = "fixed"
    BLACK_THRESH_FIXED = 100            # fixed mode: gray (0-255) below this = black. Snapshot: line 65-88, white 200+, so ~100
                                        #   Raise if the line is missed (dim light / grey tape), lower if shadows appear
    BLUR_GRAYSCALE = 7                  # box blur on gray image. Higher = smoother, loses thin lines
    BLUR_HSV = 5                        # box blur before HSV (green/red masks)
    ADAPTIVE_THRESHOLD_BLOCK_SIZE = 101 # (adaptive mode only) must be odd. ~ image height, so the threshold is close to global. Higher = more global
    ADAPTIVE_THRESHOLD_C = 60           # (adaptive mode only) pixel counts as black if this much darker than the local mean.
                                        #   With ~10-30% black in frame the cut-off lands at gray ~80-125.
                                        #   Higher = stricter (only very dark), lower = picks up grey/shadows

    # ---- [TUNE] Morphology (clean-up of masks) -----------------------
    MORPHOLOGY_KERNEL_SIZE = 5  # round kernel for black/red
    MORPHOLOGY_KERNEL_SIZE_SMALL = 3  # round kernel for green erode
    MORPHOLOGY_ITERATIONS_BLACK_ERODE_1 = 1  # removes speckle
    MORPHOLOGY_ITERATIONS_BLACK_DILATE = 6  # bridges small breaks (large = merges nearby lines)
    MORPHOLOGY_ITERATIONS_BLACK_ERODE_2 = 5  # shrinks back (dilate-erode difference = net growth)
    MORPHOLOGY_ITERATIONS_GREEN_ERODE_1 = 1
    MORPHOLOGY_ITERATIONS_GREEN_DILATE = 2
    MORPHOLOGY_ITERATIONS_GREEN_ERODE_2 = 3
    # ---- [TUNE] Minimum blob sizes (pixels^2) ------------------------
    AREA_MIN_BLACK = int(LINE_WIDTH_PX * 25)   # = 550. Black blobs smaller than ~25 px of line are ignored (noise).
                                               #   Also the shortest line stub still seen after a gap.
    AREA_MIN_GREEN = int(0.04 * (2.5 * LINE_WIDTH_PX) ** 2)  # = 272. ~4% of a full marker (after morphology), so a marker that is
                                                             #   mostly out of frame (strip ~55x19 px = 430 px^2) still counts.
                                                             #   Raise if specks/reflections are mistaken for markers.
    AREA_MIN_WHITE = 100                # white holes smaller than this inside the line are filled (~10x10 px)

    # ---- [TUNE] Intersections & green markers ------------------------
    EXPAND_EDGES = int(
        LINE_WIDTH_PX * 0.25
    )  # = 5. Half a line width would reach across the line; keep it smaller.
    # px growth of line-exit blobs when checking touching white regions
    EXPAND_WHITE_INTERSECTION = int(LINE_WIDTH_PX * 2)  # = 44. px growth of white regions to find the intersection centre
    GREEN_IGNORE_BORDER_TOUCHING = False  # True = ignore markers cut off by the image edge (center_processor default).
                                          #   False = use them; needed when the camera is close and markers are partly out of frame
    GREEN_TRACKING_DISTANCE = 15        # px: same green marker between frames if closer than this
    GREEN_IGNORE_THRESHOLD = 100        # frames a non-relevant green marker is tracked before being ignored

    # ---- [TUNE] Gaps -------------------------------------------------
    GAP_CONTOUR_MIN_ASPECT_RATIO = (
        1.5  # remaining blob must be this elongated to count as a line piece
    )
    GAP_CONTOUR_MIN_RELATIVE_SIZE = 0.12  # ... and at least this fraction of image height
    GAP_HORIZONTAL_CROP = True  # while in/after a gap only look at the centre columns
    GAP_RELATIVE_CROP_X = 0.2  # fraction cropped from left AND right during a gap
    GAP_WITH_LINE_TARGET_RELATIVE_OFFSET = (
        1  # how far ahead (x image height) the gap target is placed
    )
    GAP_START_LOCAL_RADIUS_RELATIVE = (
        0.4  # radius (x image height) of the circle fitted at the line end
    )
    GAP_START_LOCAL_MIN_PIXELS = int(
        LINE_WIDTH_PX * 5
    )  # = 110. min line pixels inside that circle
    GAP_START_LOCAL_MIN_RECT_BLACK_RATIO = 0.7  # min fill of the fitted rectangle
    GAP_START_LOCAL_RECT_BORDER_MARGIN = 2  # rectangle touching the border within this = rejected
    GAP_SEARCH_ANGLE_TOLERANCE_DEG = 30  # cone used to look for the line continuing after a gap
    GAP_SEARCH_ITERATIONS = 5  # how many times the gap search may jump to a new contour

    # ---- [TUNE] Driving / steering -----------------------------------
    VELOCITY = 380  # base forward speed
    MAX_TARGET_ANGLE = 90.0  # steering angle (deg) is clipped to +-this before the PID
    MAX_TURN = 600  # max turn command sent to drive_PID
    KP = 10.0  # PID proportional gain (turn per degree of angle)
    KI = 0.0
    KD = 0.1
    TURN_SLOWDOWN_FACTOR = 2  # larger = slows down more in tight turns
    SPIN_IN_PLACE_STRENGTH = 0.7  # turn_strength above this -> velocity 0 (spin on the spot)

    # ---- [TUNE] No-line behaviour ------------------------------------
    NO_LINE_FORWARD_FRAMES = (
        15  # frames of NO_LINE driven slowly forward before reversing (old GAP_LIMIT).
    )
    #   Camera runs at 20 fps; this counts main-loop iterations, so raise it if your
    #   loop is faster than the camera and gaps get abandoned too early.
    NO_LINE_FORWARD_SPEED_SCALE = 0.5  # fraction of VELOCITY used while creeping forward
    REVERSE_SPEED = -200  # speed used while reversing

    # ---- [TUNE] Original black mask (used for crossing + green U-turn geometry, and red removal)
    BLUR_SIZE = 9
    MORPH_CLOSE_SIZE = 7
    MORPH_OPEN_SIZE = 3
    BLACK_THRESH = 100  # gray < this = black (only for crossing / U-turn geometry). Keep equal to BLACK_THRESH_FIXED
    REMOVE_RED_FROM_BLACK = True  # original silver fix: red pixels are removed from the black mask

    # ---- [TUNE] Horizontal crossing detection (original, flag only) ---
    CROSSING_SIDE_Y_TOLERANCE = 8
    CROSSING_MIN_EDGE_RUN = 3
    CROSSING_MAX_EDGE_RUN = 45  # must exceed the line width (~33 px), more for angled lines
    CROSSING_TOP_X_TOLERANCE = 45

    # ---- [TUNE] Green U-turn detection (original) ----------------------
    HOUGH_THRESHOLD = 15
    HOUGH_MIN_LINE_LENGTH = 18
    HOUGH_MAX_LINE_GAP = 8
    GREEN_H_LOW = 35
    GREEN_H_HIGH = 90
    GREEN_S_LOW = 70
    GREEN_V_LOW = 40
    # Same HSV range is now used for the intersection green markers too (these values worked on this camera)
    THRESHOLD_GREEN = ((GREEN_H_LOW, GREEN_S_LOW, GREEN_V_LOW), (GREEN_H_HIGH, 255, 255))
    GREEN_MIN_AREA = 3000  # green square area limits (px^2) for the U-turn check. ~(2.5 x LINE_WIDTH_PX)^2 = 3025
    GREEN_MAX_AREA = 15000
    GREEN_PIXEL_THRESHOLD = 400  # cheap pre-check before the full green search
    GREEN_HSV_DOWNSAMPLE = 2
    U_TURN_COOLDOWN = 3.0  # seconds after a U-turn during which another is ignored
    U_TURN_SETTLE_TIME = 5  # seconds to wait for spin_enc to finish

    # ---- [TUNE] Rescue (red) detection (original) ----------------------
    MIN_RED = 600  # red pixels / contour area that trigger EXIT
    RED_S_MIN = 70
    RED_V_MIN = 70

    # ---- Tower -------------------------------------------------------
    TOWER_LINE_PIXELS = 2000  # dark pixels needed for lineInFrame()

    # ==================================================================
    #                    END OF TUNABLE PARAMETERS
    # ==================================================================

    def __init__(self, i2c_controller, robot, debug=False):
        self.logger = logging.getLogger("robot.line_follow")

        self.debug = debug

        self.i2c_controller = i2c_controller
        self.robot = robot
        self.camera = DownCamera()

        self.raw_frame = None
        self.cropped_frame = None

        self.startTime = time.monotonic()

        self.follow_status = Task.INIT
        self.task_started = False
        self.finished = False

        self.mode = LineModeConfig(generate_debug_frame=debug)

        # Line processor state (from center_processor)
        self._last_line_state = LineState.NO_LINE
        self._last_line_contour = None
        self._last_start_point = None
        self._last_end_point = None
        self._green_markers_memory = {}
        self._came_from_gap = False
        self.last_result = LineFollowResult()

        self.last_green_time = 0
        self.no_line_frames = 0
        self.measured_line_width = None
        self._width_log_counter = 0

        self.pid = PID(kp=self.KP, kd=self.KD, ki=self.KI)

        self.last_time = None
        self.last_error = 0
        self.debug_frame = None

        # Original kernels (black mask for crossing / U-turn, green, red)
        self.black_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.MORPH_CLOSE_SIZE, self.MORPH_CLOSE_SIZE)
        )
        self.black_open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.MORPH_OPEN_SIZE, self.MORPH_OPEN_SIZE)
        )
        self.green_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.red_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Morphology kernels
        self.MORPHOLOGY_KERNEL_ROUND = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.MORPHOLOGY_KERNEL_SIZE, self.MORPHOLOGY_KERNEL_SIZE)
        )
        self.MORPHOLOGY_KERNEL_ROUND_SMALL = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.MORPHOLOGY_KERNEL_SIZE_SMALL, self.MORPHOLOGY_KERNEL_SIZE_SMALL),
        )

    def reset(self):
        self.follow_status = Task.INIT
        self.finished = False
        self.task_started = False

    def _transition_to(self, task):
        self.logger.info(f"Line follow task: {self.follow_status.name} -> {task.name}")
        self.follow_status = task
        self.task_started = False

    def get_debug_frame(self):
        return self.debug_frame

    # ==================================================================
    # Line processing (ported from CenterLineProcessor.follow_line)
    # ==================================================================

    def process_line(self, raw_frame, cropped_frame, debug=False) -> LineFollowResult:
        if cropped_frame is None:
            return LineFollowResult(state=LineState.NO_LINE, time_stamp=time.time())

        frame = cropped_frame
        if frame.shape[1] != self.WIDTH or frame.shape[0] != self.HEIGHT:
            self.logger.error(f"Frame is not the correct size {frame.shape}")
            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)

        config = self.mode
        config.generate_debug_frame = debug

        # Original black mask: used for crossing flag and green U-turn geometry
        _, orig_black_mask = self._make_black_mask(frame)
        crossing = self._detect_horizontal_crossing(orig_black_mask)

        if debug:
            width = self._measure_line_width(orig_black_mask)
            if width is not None:
                self.measured_line_width = width
            self._width_log_counter += 1
            if self._width_log_counter % 20 == 0 and self.measured_line_width is not None:
                self.logger.info(
                    f"Measured line width ~{self.measured_line_width:.0f}px "
                    f"(LINE_WIDTH_PX={self.LINE_WIDTH_PX})"
                )

        # Original green detection -> only the both-sides U-turn is taken from it
        if self._green_present(frame):
            geometry = self._detect_junction_geometry(orig_black_mask)
            green_info = self._detect_green_squares(
                frame,
                geometry["horizontal_y"],
                geometry["vertical_x"],
                green_mask=self._get_green_mask(frame, downsample=False),
            )
            if (
                geometry["horizontal_y"] is not None
                and geometry["vertical_x"] is not None
                and green_info["green_left"]
                and green_info["green_right"]
            ):
                result = LineFollowResult(
                    state=LineState.U_TURN,
                    angle=math.pi,
                    horizontal_crossing=crossing["detected"],
                    time_stamp=time.time(),
                    debug_frame=frame.copy() if debug else None,
                )
                self.last_result = result
                return result

        result = self._follow_line(frame, config)
        result.horizontal_crossing = crossing["detected"]
        if debug and result.debug_frame is not None and crossing["detected"]:
            cv2.putText(
                result.debug_frame,
                "CROSS",
                (0, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (0, 255, 255),
                1,
            )
            if crossing["crossing_y"] is not None:
                y = int(crossing["crossing_y"])
                cv2.line(result.debug_frame, (0, y), (self.WIDTH - 1, y), (0, 255, 255), 1)
        return result

    def _follow_line(self, frame: np.ndarray, config: LineModeConfig) -> LineFollowResult:
        time_reference = time.perf_counter()

        debug_frame = frame.copy() if config.generate_debug_frame else None
        image_height, image_width = frame.shape[0], frame.shape[1]
        line_state = LineState.NO_LINE
        angle = 0
        line_angle = 0
        start_angle = 0.0
        end_pixel_count = 0
        default_start_point = (image_width // 2, image_height)
        default_end_point = (image_width // 2, 0)
        p_green = None
        p_in = None
        p_out = None

        intersection_center_relative_border = 0.4

        grayscale_frame, hsv_frame = self._preprocess_frame(frame)
        blk_binary, grn_binary = self._generate_binaries(grayscale_frame, hsv_frame)
        if self.REMOVE_RED_FROM_BLACK:
            red_mask = self._get_red_mask(frame)
            blk_binary[red_mask > 0] = 0

        blk_pixel_count = cv2.countNonZero(blk_binary)

        grn_morphed = morph_transform_image(
            grn_binary,
            self.MORPHOLOGY_ITERATIONS_GREEN_ERODE_1,
            self.MORPHOLOGY_ITERATIONS_GREEN_DILATE,
            self.MORPHOLOGY_ITERATIONS_GREEN_ERODE_2,
            self.MORPHOLOGY_KERNEL_ROUND_SMALL,
            self.MORPHOLOGY_KERNEL_ROUND,
            self.MORPHOLOGY_KERNEL_ROUND,
        )
        blk_binary = subtract_binaries(blk_binary, grn_morphed)  # subtract green from black
        blk_morphed = morph_transform_image(
            blk_binary,
            self.MORPHOLOGY_ITERATIONS_BLACK_ERODE_1,
            self.MORPHOLOGY_ITERATIONS_BLACK_DILATE,
            self.MORPHOLOGY_ITERATIONS_BLACK_ERODE_2,
            self.MORPHOLOGY_KERNEL_ROUND,
            self.MORPHOLOGY_KERNEL_ROUND,
            self.MORPHOLOGY_KERNEL_ROUND,
        )

        blk_contours_raw = find_contours(blk_morphed)
        grn_contours_raw = find_contours(grn_morphed)

        blk_contours_filtered, _ = filter_contours_by_area(blk_contours_raw, self.AREA_MIN_BLACK)
        grn_contours_filtered, _ = filter_contours_by_area(grn_contours_raw, self.AREA_MIN_GREEN)

        if config.calculate_green_center:
            green_center = self._calculate_contours_centroid(grn_contours_filtered)
            if green_center is not None:
                p_green = self._normalize_point(green_center, image_width, image_height)
                if debug_frame is not None:
                    cv2.circle(debug_frame, green_center, 3, (0, 255, 255), -1)
                    cv2.putText(
                        debug_frame,
                        "G",
                        green_center,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 255),
                        1,
                    )

        if debug_frame is not None:
            self._draw_subtle_default_points(debug_frame, default_start_point, default_end_point)

        if config.gap_crop_enabled and (
            self.GAP_HORIZONTAL_CROP
            and (
                (
                    self._last_line_state
                    in (LineState.GAP_START, LineState.GAP, LineState.GAP_WITH_LINE)
                )
                or self._came_from_gap
            )
        ):
            blk_contours_filtered = filter_contours_by_relative_area(
                blk_contours_filtered,
                grayscale_frame,
                self.GAP_RELATIVE_CROP_X,
                1 - self.GAP_RELATIVE_CROP_X,
                0,
                1,
            )

        if not blk_contours_filtered:
            self._last_line_contour = None
            line_state = LineState.NO_LINE
            if not self._came_from_gap:
                self._came_from_gap = bool(
                    self._last_line_state in (LineState.GAP_START, LineState.GAP_WITH_LINE)
                )
            self._last_line_state = line_state
            if debug_frame is not None:
                cv2.putText(
                    debug_frame,
                    line_state.name,
                    (0, 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    debug_frame,
                    f"{round((time.perf_counter() - time_reference) * 1000, 1)}ms",
                    (0, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (0, 0, 255),
                    1,
                )
            result = LineFollowResult(
                state=line_state, p_green=p_green, time_stamp=time.time(), debug_frame=debug_frame
            )
            self.last_result = result
            return result
        else:
            line_state = LineState.LINE
            self._came_from_gap = False

        line_contour_raw = self._select_line_contour(blk_contours_filtered)

        for _ in range(self.GAP_SEARCH_ITERATIONS):
            self._last_line_contour = line_contour_raw
            line_binary_raw = contour2binary(line_contour_raw, grayscale_frame)

            wht_binary = cv2.bitwise_not(line_binary_raw)
            wht_contours_raw = find_contours(wht_binary)
            wht_contours_filtered, wht_contours_too_small = filter_contours_by_area(
                wht_contours_raw, self.AREA_MIN_WHITE
            )
            cv2.drawContours(
                line_binary_raw, wht_contours_too_small, -1, 255, thickness=cv2.FILLED
            )

            if debug_frame is not None:
                self._debug_draw_contours(debug_frame, find_contours(line_binary_raw), (255, 0, 0))
                self._debug_draw_contours(debug_frame, grn_contours_filtered, (0, 255, 0))

            edge_binary = get_edge_binary(line_binary_raw)
            edge_contours_raw = find_contours(edge_binary)
            edge_contours = [ContourHandler(c, grayscale_frame) for c in edge_contours_raw]
            edge_contours_count = len(edge_contours)

            end_point = default_end_point

            start_contour, end_contour = None, None

            if edge_contours:
                grn_contours = [ContourHandler(c, grayscale_frame) for c in grn_contours_filtered if not (self.GREEN_IGNORE_BORDER_TOUCHING and contour_touches_border(grayscale_frame, c))]
                self._update_green_marker_memory(grn_contours, debug_frame=debug_frame)

                last_start = self._last_start_point
                last_end = self._last_end_point
                start_contour, end_contour = self._assign_new_start_end_contour(
                    last_start if not config.ignore_last_start else None,
                    last_end if not config.ignore_last_end else None,
                    default_start_point,
                    default_end_point,
                    edge_contours,
                )

                start_point = start_contour.center if start_contour else None

                if start_contour:
                    if debug_frame is not None:
                        cv2.circle(debug_frame, start_contour.center, 4, (255, 0, 0), -1)

                    p_in = self._normalize_point(start_contour.center, image_width, image_height)
                    start_angle = self._calculate_angle(
                        start_contour.center,
                        (
                            int(image_width * config.angle_calculation_center_x),
                            int(image_height * config.angle_calculation_center_y),
                        ),
                    )

                if end_contour:
                    if start_contour:
                        if len(edge_contours) > 2:
                            wht_contours = [
                                ContourHandler(c, grayscale_frame) for c in wht_contours_filtered
                            ]

                            for c in edge_contours:
                                c.scale_binary(self.EXPAND_EDGES)

                            intersection_with_green = False
                            if len(grn_contours) and not config.ignore_intersections_with_green:
                                adjacent_wht_countours = [
                                    c
                                    for c in wht_contours
                                    if check_overlap_binaries([
                                        start_contour.scaled_binary,
                                        c.binary,
                                    ])
                                ]

                                white_contours_with_green = 0
                                green_relevance_by_center = {
                                    grn_contour.center: False for grn_contour in grn_contours
                                }
                                for adjacent_wht_countour in adjacent_wht_countours:
                                    if debug_frame is not None:
                                        cv2.drawContours(
                                            debug_frame,
                                            [adjacent_wht_countour.contour],
                                            -1,
                                            (255, 0, 255),
                                            thickness=1,
                                        )

                                    white_has_relevant_green = False
                                    for grn_contour in grn_contours:
                                        if check_overlap_contours([
                                            grn_contour,
                                            adjacent_wht_countour,
                                        ]) and not self._check_green_ignore_from_memory(
                                            grn_contour.center
                                        ):
                                            for edge_contour in edge_contours:
                                                if (
                                                    edge_contour is not start_contour
                                                    and check_overlap_binaries([
                                                        edge_contour.scaled_binary,
                                                        adjacent_wht_countour.binary,
                                                    ])
                                                ):
                                                    end_contour = edge_contour
                                                    white_has_relevant_green = True
                                                    green_relevance_by_center[
                                                        grn_contour.center
                                                    ] = True

                                                    if debug_frame is not None:
                                                        cv2.drawContours(
                                                            debug_frame,
                                                            [adjacent_wht_countour.contour],
                                                            -1,
                                                            (0, 255, 0),
                                                            thickness=2,
                                                        )
                                                    break

                                    if white_has_relevant_green:
                                        white_contours_with_green += 1

                                for grn_contour in grn_contours:
                                    is_relevant = green_relevance_by_center.get(
                                        grn_contour.center, False
                                    )
                                    if is_relevant:
                                        self._reset_green_marker_ignore_count(grn_contour.center)
                                    else:
                                        self._increment_green_marker_ignore_count(
                                            grn_contour.center
                                        )

                                    if debug_frame is not None:
                                        cv2.drawContours(
                                            debug_frame,
                                            [grn_contour.contour],
                                            -1,
                                            (0, 255, 255) if is_relevant else (0, 0, 255),
                                            thickness=2,
                                        )
                                        ignore_count = self._green_markers_memory.get(
                                            grn_contour.center, 0
                                        )
                                        self._draw_debug_text_on_top(
                                            debug_frame,
                                            str(ignore_count),
                                            grn_contour.center,
                                            color=(255, 0, 255),
                                        )

                                if white_contours_with_green == 1:
                                    line_state = LineState.INTERSECTION_GREEN
                                    intersection_with_green = True

                            if not intersection_with_green:
                                if not config.ignore_intersections:
                                    line_state = LineState.INTERSECTION
                                    expanded_binaries = []
                                    for c in wht_contours:
                                        c.scale_binary(self.EXPAND_WHITE_INTERSECTION)
                                        expanded_binaries.append(c.scaled_binary)

                                    intersection_center = (image_width // 2, image_height // 2)
                                    intersection_contours_raw = []
                                    if expanded_binaries:
                                        intersection_binary = overlap_binaries(expanded_binaries)
                                        intersection_contours_raw = find_contours(
                                            intersection_binary
                                        )

                                    found_intersection_center = False
                                    if intersection_contours_raw:
                                        intersection_contour = ContourHandler(
                                            intersection_contours_raw[0], grayscale_frame
                                        )

                                        if not is_point_near_border(
                                            intersection_contour.center,
                                            grayscale_frame,
                                            int(
                                                image_height * intersection_center_relative_border
                                            ),
                                        ):
                                            intersection_center = intersection_contour.center
                                            found_intersection_center = True

                                        if debug_frame is not None:
                                            cv2.drawContours(
                                                debug_frame,
                                                intersection_contours_raw,
                                                -1,
                                                (0, 255, 0)
                                                if found_intersection_center
                                                else (255, 0, 0),
                                                thickness=2,
                                            )
                                            cv2.circle(
                                                debug_frame,
                                                intersection_contour.center,
                                                5,
                                                (255, 255, 0)
                                                if found_intersection_center
                                                else (255, 0, 255),
                                                -1,
                                            )

                                    theoratical_point = find_border_intersection(
                                        grayscale_frame, start_contour.center, intersection_center
                                    )
                                    if not found_intersection_center:
                                        if theoratical_point[1] > image_height / 2:
                                            if debug_frame is not None:
                                                cv2.circle(
                                                    debug_frame,
                                                    theoratical_point,
                                                    7,
                                                    (0, 128, 255),
                                                    -1,
                                                )
                                            theoratical_point = (image_width // 2, 0)
                                    closest_contour = get_closest_contour_handler(
                                        [c for c in edge_contours if c is not start_contour],
                                        theoratical_point,
                                        manhatten=False,
                                    )
                                    end_contour = closest_contour

                                    if debug_frame is not None:
                                        cv2.circle(
                                            debug_frame, theoratical_point, 3, (0, 255, 255), -1
                                        )
                                else:
                                    end_contour = get_closest_contour_handler(
                                        [c for c in edge_contours if c is not start_contour],
                                        default_end_point,
                                    )
                    else:
                        line_state = LineState.GAP_END

                    end_point = end_contour.center
                    end_pixel_count = int(end_contour.area)

                self._last_start_point = start_contour.center if start_contour else None
                self._last_end_point = end_contour.center if end_contour else None

                if start_point and end_point:
                    line_angle = self._calculate_angle(end_point, start_point)
                    p_out = self._normalize_point(end_point, image_width, image_height)
                    p_in = self._normalize_point(start_point, image_width, image_height)

            # GAP HANDLING
            gap_with_line_rect_angle_valid = False
            if not end_contour:
                if start_contour:
                    line_state = LineState.GAP_START
                else:
                    line_state = LineState.GAP_WITH_LINE

                self._last_start_point, self._last_end_point = None, None

                local_gap_result = None
                if start_contour:
                    local_gap_result = self._calculate_gap_start_target_from_local_end(
                        line_contour_raw,
                        line_binary_raw,
                        start_contour.center,
                        image_height,
                        debug_frame,
                    )

                if local_gap_result is not None and start_contour:
                    gap_anchor, end_point, gap_angle_radians, gap_rect_center = local_gap_result
                    found_contour = self._search_contour_in_direction(
                        [c for c in blk_contours_filtered if c is not line_contour_raw],
                        center=gap_anchor,
                        direction_angle=gap_angle_radians,
                        angle_tolerance_degs=self.GAP_SEARCH_ANGLE_TOLERANCE_DEG,
                        debug_frame=debug_frame,
                    )
                    if found_contour is not None:
                        self._last_line_contour = found_contour
                        line_contour_raw = found_contour
                        continue

                    line_angle = self._calculate_angle(gap_anchor, start_contour.center)

                elif not start_contour:
                    rect = cv2.minAreaRect(line_contour_raw)
                    (x, y), (h, w), box_angle_deg = rect
                    dx, dy = 0, 0

                    aspect_ratio = max(h / w, w / h) if w > 0.0 and h > 0.0 else 0.0
                    if aspect_ratio < self.GAP_CONTOUR_MIN_ASPECT_RATIO or max(w, h) < (
                        self.GAP_CONTOUR_MIN_RELATIVE_SIZE * image_height
                    ):
                        if debug_frame is not None:
                            cv2.drawContours(
                                debug_frame,
                                [np.int32(cv2.boxPoints(rect))],
                                -1,
                                (0, 0, 255),
                                thickness=1,
                            )

                    else:
                        gap_with_line_rect_angle_valid = True
                        if w > h:
                            box_angle_deg += 90

                        if debug_frame is not None:
                            cv2.drawContours(
                                debug_frame,
                                [np.int32(cv2.boxPoints(rect))],
                                -1,
                                (0, 255, 0),
                                thickness=1,
                            )

                        gap_angle_radians = math.radians(box_angle_deg)

                        found_contour = self._search_contour_in_direction(
                            [c for c in blk_contours_filtered if c is not line_contour_raw],
                            center=(int(x), int(y)),
                            direction_angle=gap_angle_radians,
                            angle_tolerance_degs=self.GAP_SEARCH_ANGLE_TOLERANCE_DEG,
                            debug_frame=debug_frame,
                        )
                        if found_contour is not None:
                            self._last_line_contour = found_contour
                            line_contour_raw = found_contour
                            continue

                        dx = -int(
                            (image_height * self.GAP_WITH_LINE_TARGET_RELATIVE_OFFSET)
                            * math.cos(math.radians(box_angle_deg))
                        )
                        dy = -int(
                            (image_height * self.GAP_WITH_LINE_TARGET_RELATIVE_OFFSET)
                            * math.sin(math.radians(box_angle_deg))
                        )

                    end_point = (int(x + dx), int(y + dy))

                else:
                    line_state = LineState.GAP
                    end_point = (start_contour.center[0], start_contour.center[1] - image_height)

                    if debug_frame is not None:
                        cv2.putText(
                            debug_frame,
                            "gap start local rect rejected",
                            (0, 52),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.25,
                            (0, 0, 255),
                            1,
                        )

            if line_state is LineState.GAP_WITH_LINE and not gap_with_line_rect_angle_valid:
                angle = 0
            else:
                angle = self._calculate_angle(
                    end_point,
                    (
                        int(image_width * config.angle_calculation_center_x),
                        int(image_height * config.angle_calculation_center_y),
                    ),
                )
                if line_state is LineState.GAP_WITH_LINE and abs(math.degrees(angle)) > 100.0:
                    angle = self._flip_angle_180_rad(angle)
            p_out = self._normalize_point(end_point, image_width, image_height)

            if debug_frame is not None:
                line_start, line_end = clip_line(
                    debug_frame,
                    (
                        int(image_width * config.angle_calculation_center_x),
                        int(image_height * config.angle_calculation_center_y),
                    ),
                    end_point,
                )
                cv2.line(debug_frame, line_start, line_end, (0, 0, 255), thickness=1)
                if is_point_in_image(end_point, debug_frame):
                    cv2.circle(debug_frame, end_point, 4, (0, 0, 255), -1)
                cv2.putText(
                    debug_frame,
                    line_state.name,
                    (0, 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    debug_frame,
                    f"{round(math.degrees(angle))}deg",
                    (0, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    debug_frame,
                    f"b: {blk_pixel_count}px",
                    (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    debug_frame,
                    f"e: {end_pixel_count}px",
                    (0, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    debug_frame,
                    f"{round((time.perf_counter() - time_reference) * 1000, 1)}ms",
                    (0, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (0, 0, 255),
                    1,
                )

            self._last_line_state = line_state
            result = LineFollowResult(
                state=line_state,
                angle=angle,
                line_angle=line_angle,
                start_angle=start_angle,
                black_pixel_count=blk_pixel_count,
                end_pixel_count=end_pixel_count,
                edge_contour_count=edge_contours_count,
                start_pixel_count=int(start_contour.area) if start_contour else 0,
                p_in=p_in if start_contour else None,
                p_out=p_out,
                p_green=p_green,
                time_stamp=time.time(),
                debug_frame=debug_frame,
            )
            self.last_result = result
            return result

        # The gap search kept jumping to new contours and used up all iterations.
        # (center_processor falls off the end and returns None here; we return NO_LINE instead.)
        self._last_line_state = LineState.NO_LINE
        result = LineFollowResult(
            state=LineState.NO_LINE,
            p_green=p_green,
            time_stamp=time.time(),
            debug_frame=debug_frame,
        )
        self.last_result = result
        return result

    def _search_contour_in_direction(
        self, blk_contours, center, direction_angle, angle_tolerance_degs, debug_frame=None
    ):
        best_contour = None
        best_distance = 0.0

        angle_tolerance = math.radians(angle_tolerance_degs)

        cx, cy = center

        for contour in blk_contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            x_c = int(M["m10"] / M["m00"])
            y_c = int(M["m01"] / M["m00"])

            dx = x_c - cx
            dy = y_c - cy

            angle_to_contour = math.atan2(dy, dx)
            angle_diff = self._normalize_angle_diff(angle_to_contour - direction_angle)

            if abs(angle_diff) <= angle_tolerance:
                dist = math.sqrt(dx * dx + dy * dy)

                # Keep taking the furthest matching contour.
                if dist > best_distance:
                    best_distance = dist
                    best_contour = contour

                    if debug_frame is not None:
                        cv2.circle(debug_frame, (x_c, y_c), 5, (0, 255, 0), -1)

            elif debug_frame is not None:
                cv2.drawContours(debug_frame, [contour], -1, (0, 0, 128), thickness=1)
                cv2.putText(
                    debug_frame,
                    str(round(math.degrees(angle_diff))),
                    (min(x_c - 10, debug_frame.shape[1] - 1), y_c),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 255),
                    2,
                )

        if debug_frame is not None:
            end_pt = (
                int(cx + 100 * math.cos(direction_angle)),
                int(cy + 100 * math.sin(direction_angle)),
            )
            cv2.circle(debug_frame, (cx, cy), 5, (255, 100, 0), -1)
            cv2.line(debug_frame, (cx, cy), end_pt, (255, 100, 0), 1)

        return best_contour

    def _calculate_gap_start_target_from_local_end(
        self, line_contour, line_binary, start_point, image_height, debug_frame=None
    ):
        contour_points = line_contour.reshape(-1, 2)
        if contour_points.size == 0:
            return None

        start = np.array(start_point, dtype=np.float32)
        deltas = contour_points.astype(np.float32) - start
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))

        max_distance = float(np.max(distances))
        end_band_px = max(3.0, image_height * 0.05)

        end_candidates = contour_points[distances >= max_distance - end_band_px]
        gap_anchor_raw = np.mean(end_candidates, axis=0)
        gap_anchor = (int(gap_anchor_raw[0]), int(gap_anchor_raw[1]))

        radius = max(1, int(image_height * self.GAP_START_LOCAL_RADIUS_RELATIVE))

        local_mask = np.zeros(line_binary.shape, dtype=np.uint8)
        cv2.circle(local_mask, gap_anchor, radius, 255, thickness=cv2.FILLED)

        local_binary = cv2.bitwise_and(line_binary, local_mask)

        local_y, local_x = np.where(local_binary == 255)
        if len(local_x) < self.GAP_START_LOCAL_MIN_PIXELS:
            if debug_frame is not None:
                cv2.circle(debug_frame, gap_anchor, radius, (0, 0, 255), 1)
                cv2.circle(debug_frame, gap_anchor, 4, (0, 0, 255), -1)
            return None

        local_points = np.column_stack((local_x, local_y)).astype(np.float32)

        local_rect = cv2.minAreaRect(local_points)
        (_, _), (rect_w, rect_h), _ = local_rect

        if rect_w <= 0 or rect_h <= 0:
            return None

        rect_box = np.int32(cv2.boxPoints(local_rect))
        rect_center = (int(local_rect[0][0]), int(local_rect[0][1]))
        image_width = line_binary.shape[1]
        border_margin = self.GAP_START_LOCAL_RECT_BORDER_MARGIN
        touches_border = bool(
            np.any(rect_box[:, 0] <= border_margin)
            or np.any(rect_box[:, 0] >= image_width - 1 - border_margin)
            or np.any(rect_box[:, 1] <= border_margin)
            or np.any(rect_box[:, 1] >= image_height - 1 - border_margin)
        )
        if touches_border:
            if debug_frame is not None:
                cv2.circle(debug_frame, gap_anchor, radius, (0, 0, 255), 1)
                cv2.circle(debug_frame, gap_anchor, 4, (0, 0, 255), -1)
                cv2.drawContours(debug_frame, [rect_box], -1, (0, 0, 255), thickness=1)
                self._draw_debug_text_on_top(
                    debug_frame, "gap border", gap_anchor, color=(0, 0, 255)
                )
            return None

        rect_mask = np.zeros(line_binary.shape, dtype=np.uint8)
        cv2.drawContours(rect_mask, [rect_box], -1, 255, thickness=cv2.FILLED)
        rect_area = cv2.countNonZero(rect_mask)
        rect_black_ratio = 0.0
        if rect_area > 0:
            rect_black_ratio = (
                cv2.countNonZero(cv2.bitwise_and(line_binary, rect_mask)) / rect_area
            )
        if rect_black_ratio < self.GAP_START_LOCAL_MIN_RECT_BLACK_RATIO:
            if debug_frame is not None:
                cv2.circle(debug_frame, gap_anchor, radius, (0, 0, 255), 1)
                cv2.circle(debug_frame, gap_anchor, 4, (0, 0, 255), -1)
                cv2.drawContours(debug_frame, [rect_box], -1, (0, 0, 255), thickness=1)
                self._draw_debug_text_on_top(
                    debug_frame, f"gap fill {rect_black_ratio:.2f}", gap_anchor, color=(0, 0, 255)
                )
            return None

        aspect_ratio = max(rect_w / rect_h, rect_h / rect_w)
        if aspect_ratio < self.GAP_CONTOUR_MIN_ASPECT_RATIO or min(rect_w, rect_h) < (
            self.GAP_CONTOUR_MIN_RELATIVE_SIZE * image_height
        ):
            if debug_frame is not None:
                cv2.circle(debug_frame, gap_anchor, radius, (0, 0, 255), 1)
                cv2.circle(debug_frame, gap_anchor, 4, (0, 0, 255), -1)

                local_contours = find_contours(local_binary)
                if local_contours:
                    cv2.drawContours(debug_frame, local_contours, -1, (0, 0, 255), thickness=1)

                cv2.drawContours(debug_frame, [rect_box], -1, (0, 0, 255), thickness=1)
                self._draw_debug_text_on_top(
                    debug_frame, f"gap aspect {aspect_ratio:.1f}", gap_anchor, color=(0, 0, 255)
                )
            return None

        vx, vy, _, _ = cv2.fitLine(local_points, cv2.DIST_L2, 0, 0.01, 0.01)

        direction = np.array([float(vx[0]), float(vy[0])], dtype=np.float32)

        anchor_vector = np.array(gap_anchor, dtype=np.float32) - start
        anchor_norm = float(np.linalg.norm(anchor_vector))

        if anchor_norm > 1e-6:
            anchor_direction = anchor_vector / anchor_norm

            if float(np.dot(direction, anchor_direction)) < 0:
                direction *= -1

        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            return None

        direction /= norm

        target_distance = image_height * self.GAP_WITH_LINE_TARGET_RELATIVE_OFFSET
        target = np.array(gap_anchor, dtype=np.float32) + direction * target_distance

        end_point = (int(target[0]), int(target[1]))

        fit_angle_radians = math.atan2(float(direction[1]), float(direction[0]))
        gap_angle_radians = fit_angle_radians

        if debug_frame is not None:
            cv2.circle(debug_frame, gap_anchor, radius, (255, 255, 0), 1)
            cv2.circle(debug_frame, gap_anchor, 4, (255, 0, 255), -1)

            local_contours = find_contours(local_binary)
            if local_contours:
                cv2.drawContours(debug_frame, local_contours, -1, (255, 255, 0), thickness=1)
                cv2.drawContours(debug_frame, [rect_box], -1, (0, 255, 255), thickness=1)

            line_length = int(target_distance)

            fit_start = (
                int(gap_anchor[0] - direction[0] * line_length),
                int(gap_anchor[1] - direction[1] * line_length),
            )
            fit_end = (
                int(gap_anchor[0] + direction[0] * line_length),
                int(gap_anchor[1] + direction[1] * line_length),
            )

            fit_start, fit_end = clip_line(debug_frame, fit_start, fit_end)

            cv2.line(debug_frame, fit_start, fit_end, (255, 0, 255), thickness=1)

            self._draw_debug_text_on_top(
                debug_frame,
                f"fit {math.degrees(fit_angle_radians):.0f}",
                (max(0, gap_anchor[0] - 20), max(10, gap_anchor[1] - 8)),
                color=(255, 0, 255),
            )

        return gap_anchor, end_point, gap_angle_radians, rect_center

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess_frame(self, frame):
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_grayscale_frame = cv2.boxFilter(
            grayscale_frame, -1, (self.BLUR_GRAYSCALE, self.BLUR_GRAYSCALE)
        )
        blurred_rgb_frame = cv2.boxFilter(frame, -1, (self.BLUR_HSV, self.BLUR_HSV))
        hsv_frame = cv2.cvtColor(blurred_rgb_frame, cv2.COLOR_BGR2HSV)
        return blurred_grayscale_frame, hsv_frame

    def _generate_binaries(self, grayscale_frame, hsv_frame):
        if self.BLACK_THRESHOLD_MODE == "fixed":
            _, blk_binary = cv2.threshold(
                grayscale_frame, self.BLACK_THRESH_FIXED, 255, cv2.THRESH_BINARY_INV
            )
        else:
            blk_binary = adaptive_threshold(
                grayscale_frame,
                self.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
                self.ADAPTIVE_THRESHOLD_C,
                True,
            )
        grn_binary = cv2.inRange(hsv_frame, self.THRESHOLD_GREEN[0], self.THRESHOLD_GREEN[1])
        return blk_binary, grn_binary

    # ------------------------------------------------------------------
    # Contour / start-end selection
    # ------------------------------------------------------------------

    def _select_line_contour(self, blk_contours):
        if self._last_line_contour is not None:
            return get_closest_contour_to_contour(blk_contours, self._last_line_contour)
        return get_largest_contour(blk_contours)

    def _assign_new_start_end_contour(
        self, last_start, last_end, default_start, default_end, edge_contours
    ):
        if len(edge_contours) == 0:
            return None, None
        elif len(edge_contours) == 1:
            contour = edge_contours[0]
            start = default_start
            end = default_end
            if last_start and last_end:
                start = last_start
                end = last_end
            distance_to_start = calculate_manhattan_distance(start, contour.center)
            distance_to_end = calculate_manhattan_distance(end, contour.center)
            return (contour, None) if distance_to_start < distance_to_end else (None, contour)
        else:
            min_start, min_end = None, None
            min_start_dist, min_end_dist = float("inf"), float("inf")
            second_best_start, second_best_end = None, None
            second_best_start_dist, second_best_end_dist = float("inf"), float("inf")
            for contour in edge_contours:
                if last_start is None and last_end is None:
                    start_dist = calculate_distance_between_points(default_start, contour.center)
                    end_dist = calculate_distance_between_points(default_end, contour.center)
                else:
                    start_dist = (
                        calculate_manhattan_distance(last_start, contour.center)
                        if last_start
                        else 10000.0
                    )
                    end_dist = (
                        calculate_manhattan_distance(last_end, contour.center)
                        if last_end
                        else 10000.0
                    )
                if start_dist < min_start_dist:
                    second_best_start, second_best_start_dist = min_start, min_start_dist
                    min_start, min_start_dist = contour, start_dist
                elif start_dist < second_best_start_dist:
                    second_best_start, second_best_start_dist = contour, start_dist

                if end_dist < min_end_dist:
                    second_best_end, second_best_end_dist = min_end, min_end_dist
                    min_end, min_end_dist = contour, end_dist
                elif end_dist < second_best_end_dist:
                    second_best_end, second_best_end_dist = contour, end_dist

            if min_start == min_end:
                if min_start_dist > min_end_dist:
                    min_start = second_best_start
                else:
                    min_end = second_best_end
            return min_start, min_end

    # ------------------------------------------------------------------
    # ORIGINAL code restored: black mask, red/rescue, horizontal crossing,
    # green U-turn detection (unchanged from the previous version)
    # ------------------------------------------------------------------

    def _make_black_mask(self, frame):
        """Create a mask of dark regions. Red regions are removed so silver isn't included."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        gray = cv2.GaussianBlur(gray, (self.BLUR_SIZE, self.BLUR_SIZE), 0)

        _, black_mask = cv2.threshold(gray, self.BLACK_THRESH, 255, cv2.THRESH_BINARY_INV)

        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, self.black_close_kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, self.black_open_kernel)

        # Remove red pixels
        red_mask = self._get_red_mask(frame)

        if red_mask is not None:
            black_mask[red_mask > 0] = 0

        return gray, black_mask

    def _get_red_mask(self, image):
        if image is None:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(
            hsv,
            np.array([0, self.RED_S_MIN, self.RED_V_MIN], dtype=np.uint8),
            np.array([10, 255, 255], dtype=np.uint8),
        )

        mask2 = cv2.inRange(
            hsv,
            np.array([170, self.RED_S_MIN, self.RED_V_MIN], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        )

        red_mask = mask1 | mask2

        # Close small gaps in reflected red areas
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.red_kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.red_kernel)

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
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= self.MIN_RED:
                self.logger.info(f"Red area: {area:.1f}")
                return True

        return False

    def _measure_line_width(self, black_mask):
        """Median width of the black line in the lower rows (rows with exactly one run only,
        so crossings/gaps don't distort it). Debug aid for setting LINE_WIDTH_PX."""
        h, w = black_mask.shape
        widths = []
        for y in range(int(h * 0.6), h, 3):
            row = (black_mask[y] > 0).astype(np.int8)
            d = np.diff(np.concatenate(([0], row, [0])))
            runs = np.where(d == -1)[0] - np.where(d == 1)[0]
            runs = runs[(runs >= 3) & (runs <= w * 0.35)]
            if len(runs) == 1:
                widths.append(int(runs[0]))
        return float(np.median(widths)) if len(widths) >= 5 else None

    def _find_edge_run(self, black_mask, edge):
        """Find the longest compact black run touching one image edge."""
        h, w = black_mask.shape

        if edge == "left":
            values = black_mask[:, 0] > 0
        elif edge == "right":
            values = black_mask[:, w - 1] > 0
        elif edge == "top":
            values = black_mask[0, :] > 0
        else:
            raise ValueError(f"Unknown edge: {edge}")

        none = {"detected": False, "start": None, "end": None, "centre": None, "length": 0}

        indices = np.where(values)[0]

        if len(indices) == 0:
            return none

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
            return none

        length, start, end = max(valid_runs, key=lambda x: x[0])

        return {
            "detected": True,
            "start": start,
            "end": end,
            "centre": (start + end) / 2.0,
            "length": length,
        }

    def _detect_horizontal_crossing(self, black_mask):
        """Black exiting left, right and top edges, sides at the same Y, top near centre."""
        left = self._find_edge_run(black_mask, "left")
        right = self._find_edge_run(black_mask, "right")
        top = self._find_edge_run(black_mask, "top")

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

        if not left["detected"] or not right["detected"] or not top["detected"]:
            return result

        if abs(left["centre"] - right["centre"]) > self.CROSSING_SIDE_Y_TOLERANCE:
            return result

        if abs(top["centre"] - self.WIDTH / 2.0) > self.CROSSING_TOP_X_TOLERANCE:
            return result

        result["detected"] = True
        result["crossing_y"] = (left["centre"] + right["centre"]) / 2.0

        return result

    def _get_green_mask(self, frame, downsample=False):
        if frame is None:
            return None

        if downsample:
            frame = frame[:: self.GREEN_HSV_DOWNSAMPLE, :: self.GREEN_HSV_DOWNSAMPLE]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array(
            [self.GREEN_H_LOW, self.GREEN_S_LOW, self.GREEN_V_LOW], dtype=np.uint8
        )
        upper_green = np.array([self.GREEN_H_HIGH, 255, 255], dtype=np.uint8)

        return cv2.inRange(hsv, lower_green, upper_green)

    def _green_present(self, frame):
        """Cheap green test before running the more expensive contour / Hough processing."""
        if frame is None:
            return False

        green_mask = self._get_green_mask(frame, downsample=True)

        if green_mask is None:
            return False

        return cv2.countNonZero(green_mask) >= self.GREEN_PIXEL_THRESHOLD

    def _detect_junction_geometry(self, black_mask):
        """Use Hough lines for the green-marker geometry."""
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

                if abs(dy) < length * 0.25:
                    horizontal.append((length, x1, y1, x2, y2))

                elif abs(dx) < length * 0.25:
                    vertical.append((length, x1, y1, x2, y2))

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

        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, self.green_kernel)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centres = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.GREEN_MIN_AREA or area > self.GREEN_MAX_AREA:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

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

            centres.append((x + w / 2.0, y + h / 2.0, area))

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

    # ------------------------------------------------------------------
    # Angle / geometry helpers
    # ------------------------------------------------------------------

    def _calculate_angle(self, point, reference_point):
        dx = reference_point[0] - point[0]
        dy = reference_point[1] - point[1]
        angle = math.atan2(dy, dx)
        return self._adjust_angle(angle)

    def _adjust_angle(self, angle):
        adjusted_angle = angle - 0.5 * math.pi
        if adjusted_angle < -math.pi:
            adjusted_angle += 2 * math.pi
        elif adjusted_angle > math.pi:
            adjusted_angle -= 2 * math.pi
        return adjusted_angle

    def _normalize_angle_diff(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    def _normalize_angle_rad(self, angle):
        return ((float(angle) + math.pi) % (2 * math.pi)) - math.pi

    def _flip_angle_180_rad(self, angle):
        return self._normalize_angle_rad(float(angle) + math.pi)

    def _normalize_point(self, point, image_width, image_height):
        return (point[0] / image_width, point[1] / image_height)

    def _calculate_contours_centroid(self, contours):
        if not contours:
            return None

        m00_total = 0.0
        m10_total = 0.0
        m01_total = 0.0
        fallback_points = []

        for contour in contours:
            m = cv2.moments(contour)
            m00 = m.get("m00", 0.0)
            if m00 > 0:
                m00_total += m00
                m10_total += m["m10"]
                m01_total += m["m01"]
            else:
                x, y, w, h = cv2.boundingRect(contour)
                fallback_points.append((x + w // 2, y + h // 2))

        if m00_total > 0:
            return (int(m10_total / m00_total), int(m01_total / m00_total))

        if fallback_points:
            x_mean = int(sum(p[0] for p in fallback_points) / len(fallback_points))
            y_mean = int(sum(p[1] for p in fallback_points) / len(fallback_points))
            return (x_mean, y_mean)

        return None

    # ------------------------------------------------------------------
    # Green marker memory
    # ------------------------------------------------------------------

    def _update_green_marker_memory(self, green_contours, debug_frame=None):
        if not green_contours:
            return
        old_markers = self._green_markers_memory
        self._green_markers_memory = {}
        for green_contour in green_contours:
            known_marker = False
            for center, ignore_count in old_markers.items():
                if known_marker:
                    break
                if (
                    calculate_distance_between_points(green_contour.center, center)
                    < self.GREEN_TRACKING_DISTANCE
                ):
                    self._green_markers_memory[green_contour.center] = ignore_count
                    known_marker = True
            if not known_marker:
                self._green_markers_memory[green_contour.center] = 0

    def _set_all_green_ignore_values(self, value):
        for key in self._green_markers_memory:
            self._green_markers_memory[key] = value

    def _increment_green_marker_ignore_count(self, green_center):
        if green_center in self._green_markers_memory:
            self._green_markers_memory[green_center] += 1

    def _reset_green_marker_ignore_count(self, green_center):
        if green_center in self._green_markers_memory:
            self._green_markers_memory[green_center] = 0

    def _check_green_ignore_from_memory(self, green_center):
        if green_center in self._green_markers_memory:
            return self._green_markers_memory[green_center] > self.GREEN_IGNORE_THRESHOLD
        return False

    # ------------------------------------------------------------------
    # Debug drawing helpers
    # ------------------------------------------------------------------

    def _draw_debug_text_on_top(
        self, debug_frame, text, position, color=(255, 255, 255), scale=0.3, outline=2
    ):
        pos = (int(position[0]), int(position[1]))
        cv2.putText(
            debug_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), outline + 1
        )
        cv2.putText(debug_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

    def _draw_subtle_default_points(self, debug_frame, start_point, end_point):
        cv2.circle(
            debug_frame,
            (int(start_point[0]), min(int(start_point[1]), debug_frame.shape[0] - 1)),
            3,
            (120, 120, 120),
            -1,
        )
        cv2.circle(debug_frame, (int(end_point[0]), int(end_point[1])), 3, (120, 120, 120), -1)

    def _debug_draw_contours(self, debug_frame, contours, color):
        cv2.drawContours(debug_frame, contours, -1, color, thickness=1)

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset_start_end_point(self):
        self._last_start_point = None
        self._last_end_point = None

    def reset_line_state(self):
        self._last_line_state = LineState.NO_LINE
        self._last_line_contour = None
        self._green_markers_memory = {}
        self._came_from_gap = False
        self.last_result = LineFollowResult()
        self.reset_start_end_point()

    def _reset_line_tracking(self):
        self.pid.reset()
        self.last_time = None
        self.last_error = 0
        self.no_line_frames = 0
        self.reset_line_state()

    # ==================================================================
    # Water tower
    # ==================================================================

    def lineInFrame(self) -> bool:
        frame = self.cropped_frame

        if frame is None:
            return False

        line = cv2.inRange(frame, (0, 0, 0), (60, 60, 60))

        return cv2.countNonZero(line) > self.TOWER_LINE_PIXELS

    # ==================================================================
    # Main loop
    # ==================================================================

    def main(self):
        now = time.monotonic()
        if self.last_time is None:
            dt = 0.0
        else:
            dt = now - self.last_time

        self.last_time = now

        frames = self.camera.get_frame()
        if frames is None:  # DownCamera returns None until the first frame has arrived
            return
        self.raw_frame, self.cropped_frame = frames[0], frames[1]
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

                self.robot.spin_enc(80, 550)
                time.sleep(3)
                self.robot.drive_PID(600, -490)
                time.sleep(3.6)
                self.robot.drive_PID(200)

            if self.lineInFrame():
                self.robot.stop_moving()
                self._transition_to(Task.FOLLOW)

        elif self.follow_status == Task.FOLLOW:
            if not self.task_started:
                self._reset_line_tracking()
                self.task_started = True

            # Rescue detection before line processing (original)
            if self.red_detected(c_frame):
                self.logger.info("Red detected")
                self._transition_to(Task.EXIT)
                return

            result = self.process_line(r_frame, c_frame, debug=self.debug)

            if self.debug and result.debug_frame is not None:
                self.debug_frame = result.debug_frame

            state = result.state

            if state == LineState.U_TURN:
                # Original U-turn handling: green on both sides
                if (
                    self.last_green_time
                    < time.monotonic()
                    < self.last_green_time + self.U_TURN_COOLDOWN
                ):
                    self.logger.info("U-turn detected within cooldown of last green turn")
                    self.robot.drive_PID(200)
                    time.sleep(0.1)
                    return
                self.logger.info("U-turn detected")

                self.robot.spin_enc(180.0)
                time.sleep(self.U_TURN_SETTLE_TIME)
                self._reset_line_tracking()
                self.last_green_time = time.monotonic()

            elif state == LineState.NO_LINE:
                self.no_line_frames += 1
                if self.no_line_frames <= self.NO_LINE_FORWARD_FRAMES:
                    self.logger.info("No line, moving forward")
                    self.robot.drive_PID(int(self.VELOCITY * self.NO_LINE_FORWARD_SPEED_SCALE), 0)
                else:
                    self.logger.info("No line, reversing")
                    self.robot.drive_PID(self.REVERSE_SPEED)

            else:
                # LINE, INTERSECTION, INTERSECTION_GREEN, GAP_START, GAP, GAP_WITH_LINE, GAP_END:
                # all of them give a steering angle pointing at the next piece of line.
                self.no_line_frames = 0

                target_angle = float(
                    np.clip(
                        math.degrees(result.angle),
                        -self.MAX_TARGET_ANGLE,
                        self.MAX_TARGET_ANGLE,
                    )
                )

                error_pid = self.pid.update(target_angle, dt)

                turn_error = np.clip(error_pid, -self.MAX_TURN, self.MAX_TURN)

                turn_strength = abs(turn_error) / self.MAX_TURN

                # Slow down on large steering errors
                speed_scale = 1.0 / (1.0 + self.TURN_SLOWDOWN_FACTOR * turn_strength**2)
                velocity = self.VELOCITY * speed_scale

                if self.debug:
                    self.logger.info(
                        f"{state.name} angle={target_angle:.1f} PID Error: {turn_error}, "
                        f"Scaled Velocity: {velocity} due to speed scaling {speed_scale}"
                    )

                # At very large steering errors, turn on the spot
                if turn_strength > self.SPIN_IN_PLACE_STRENGTH:
                    velocity = 0

                self.robot.drive_PID(int(velocity), int(turn_error))

                self.last_error = turn_error

        elif self.follow_status == Task.INIT:
            if not self.task_started:
                self.task_started = True

                self.raw_frame = None
                self.cropped_frame = None

                self.startTime = time.monotonic()

                self.finished = False

                self.no_line_frames = 0
                self.last_green_time = 0
                self.last_time = None
                self.last_error = 0
                self.debug_frame = None

                self.reset_line_state()
                self.pid.reset()

                self._transition_to(Task.FOLLOW)

        elif self.follow_status == Task.EXIT:
            if not self.task_started:
                self.task_started = True
                self.raw_frame = None
                self.cropped_frame = None
                self.finished = True

    def is_finished(self):
        return self.finished


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
        self.integral = np.clip(self.integral, -1000.0, 1000.0)

        derivative = (error - self.previous_error) / dt

        self.previous_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.initialised = False
