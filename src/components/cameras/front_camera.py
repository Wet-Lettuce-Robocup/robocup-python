import cv2
from picamera2 import Picamera2
from picamera2.utils import Transform


class FrontCamera:
    def __init__(self):
        self.cam = Picamera2()
        self._init_camera()

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

    def _crop_frame(self, frame):
        height, width = frame.shape[:2]

        bottom_margin = int(height * 0.02)
        start_y = int(height / 3)
        end_y = int(height - bottom_margin)

        side_margin = int(width * 0.05)
        start_x = side_margin
        end_x = width - side_margin

        return frame[start_y:end_y, start_x:end_x]

    def get_frame(self):
        raw_frame = self.cam.get_frame()
        cropped_frame = self._crop_frame(raw_frame)
        return raw_frame, cropped_frame
