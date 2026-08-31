import threading

import cv2
import numpy as np
import yaml
from picamera2 import Picamera2
from picamera2.utils import Transform


class FrontCamera:
    def __init__(self):
        self.cam = Picamera2(0)
        self._init_camera()

        with open("src/components/cameras/ost.yaml", "r") as f:
            calib_data = yaml.safe_load(f)
        raw_matrix = calib_data["camera_matrix"]

        raw_dist = calib_data["distortion_coefficients"]

        self.camera_matrix = np.array(raw_matrix["data"], dtype=np.float32).reshape(
            raw_matrix["rows"], raw_matrix["cols"]
        )
        self.distortion_coefficients = np.array(raw_dist["data"], dtype=np.float32).reshape(
            raw_dist["rows"], raw_dist["cols"]
        )

        self.frame = None

    def _init_camera(self):
        self.cam.configure(
            self.cam.create_video_configuration(
                sensor={"output_size": (2304, 1296)},  # 16:9 aspect ratio
                main={
                    "format": "RGB888",
                    "size": (1536, 864),
                },
                controls={"FrameRate": 20},
                transform=Transform(hflip=True, vflip=True),  # Fixes 180 degree rotation
            )
        )
        self.cam.set_controls({"AfMode": 2})
        self.cam.start()

        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()

    def _crop_frame(self, frame):
        height, width = frame.shape[:2]

        bottom_margin = int(height * 0.02)
        start_y = int(height / 3)
        end_y = int(height - bottom_margin)

        side_margin = int(width * 0.05)
        start_x = side_margin
        end_x = width - side_margin

        return (frame[start_y:end_y, start_x:end_x]), (start_x, start_y), (end_x - 1, end_y - 1)

    def _undistort_frame(self, frame):
        undistorted_frame = cv2.undistort(
            frame,
            self.camera_matrix,
            self.distortion_coefficients,
            None,
            self.camera_matrix,
        )

        return undistorted_frame

    def update(self):
        self.frame = self.cam.capture_array()

    def get_frame(self, debug=False):
        raw_frame = self.frame
        if raw_frame is None:
            return
        undist_frame = self._undistort_frame(raw_frame)
        cropped_frame, debug_top_left, debug_bottom_right = self._crop_frame(undist_frame)

        if debug:
            debug_frame = raw_frame.copy()
            cv2.rectangle(debug_frame, debug_top_left, debug_bottom_right, (0, 0, 255), 2)
            cv2.imshow("Frame", debug_frame)
            cv2.waitKey(1)
            return raw_frame, cropped_frame, debug_top_left, debug_bottom_right
        else:
            return raw_frame, cropped_frame
