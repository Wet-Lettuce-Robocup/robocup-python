import threading

import cv2
from picamera2 import Picamera2


class DownCamera:
    def __init__(self):
        self.cam = Picamera2(1)

        self.frame = None
        self._running = True
        self._frame_lock = threading.Lock()
        self.thread = None

        self._init_camera()

    def _init_camera(self):
        self.cam.configure(
            self.cam.create_video_configuration(
                sensor={"output_size": (2304, 1296)},  # 16:9 aspect ratio
                main={
                    "format": "RGB888",
                    "size": (240, 135),
                },
                controls={"FrameRate": 20},
            )
        )
        self.cam.set_controls({"AfMode": 2})
        self.cam.start()

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _crop_frame(self, frame):
        # height, width = frame.shape[:2]

        # bottom_margin = int(height * 0.02)
        # start_y = int(height / 3)
        # end_y = int(height - bottom_margin)

        # side_margin = int(width * 0.05)
        # start_x = side_margin
        # end_x = width - side_margin

        # return (frame[start_y:end_y, start_x:end_x]), (start_x, start_y), (end_x - 1, end_y - 1)
        return frame, (0, 0), (0, 0)  # Temporarily until crop is calibrated

    def update(self):
        while self._running:
            frame = self.cam.capture_array()

            with self._frame_lock:
                self.frame = frame

    def get_frame(self, debug=False):
        with self._frame_lock:
            if self.frame is None:
                return None

            raw_frame = self.frame.copy()

        cropped_frame, debug_top_left, debug_bottom_right = self._crop_frame(raw_frame)

        if debug:
            debug_frame = raw_frame.copy()
            cv2.rectangle(debug_frame, debug_top_left, debug_bottom_right, (0, 0, 255), 2)
            cv2.imshow("Frame", debug_frame)
            cv2.waitKey(1)
            return raw_frame, cropped_frame, debug_top_left, debug_bottom_right
        else:
            return raw_frame, cropped_frame

    def close(self):
        self._running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        self.cam.stop()
        self.cam.close()
