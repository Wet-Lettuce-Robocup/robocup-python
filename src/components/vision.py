import logging
import math

import cv2
from ultralytics import YOLO

from src.components.cameras.front_camera import FrontCamera


class Vision:
    """Handles camera capture and YOLO object detection."""

    conf_threshold = 0.7

    # Raw camera frame size
    dw = 1536
    dh = 864

    # Camera calibration values
    fx = 683.31285
    fy = 683.10689
    cx = 764.89803
    cy = 408.4118

    # Known object dimensions (metres) for position estimation
    ball_diameter = 0.05
    evac_height = 0.06

    def __init__(self, debug=True):
        self.logger = logging.getLogger("vision")

        self.camera = FrontCamera()

        self.model = YOLO("src/components/cameras/state.pt")

        self.debug = debug

        # Crop parameters
        self.start_x = int(self.dw * 0.05)
        self.start_y = int(self.dh / 3)

    def _crop_box_to_frame(self, x1, y1, x2, y2):
        return (
            int(x1 + self.start_x),
            int(y1 + self.start_y),
            int(x2 + self.start_x),
            int(y2 + self.start_y),
        )

    def get_all_objects(self):

        raw_frame, cropped_frame = self.camera.get_frame()
        if raw_frame is None:
            self.logger.info("No frame given received from front camera")
            return None

        results = self.model.predict(
            cropped_frame, conf=self.conf_threshold, stream=True, imgsz=640, verbose=False
        )

        all_detections = {}
        counts = {"silver": 0, "black": 0, "green": 0, "red": 0}

        for i in results:
            if self.debug:
                annotated_frame = i.plot()
                cv2.imshow("a", annotated_frame)
                cv2.waitKey(1)
                # self.out.write(annotated_frame)

            for x1, y1, x2, y2, conf, cls in i.boxes.data.tolist():
                x1, y1, x2, y2 = self._crop_box_to_frame(x1, y1, x2, y2)

                class_name = self.model.names[int(cls)]

                if class_name not in counts:
                    self.logger.warning(f"Unknown class detected: {class_name}")

                data = {
                    "cls": class_name,
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "conf": float(conf),
                }

                counts[class_name] += 1
                all_detections.update({len(all_detections): data})

        return {"detections": all_detections, "counts": counts}

    def filter_objects(self, objects, target):
        """target can only be "ball" or "evac_point"."""

        if objects is None:
            self.logger.warning("No objects to filter")
            return {}
        if target not in ["ball", "evac_point"]:
            self.logger.warning("Invalid target set for filtering")

        filtered_detections = {}

        for data in objects.values():
            if (target == "ball" and data["cls"] == "silver") or (
                target == "evac_point" and data["cls"] == "green"
            ):
                filtered_detections.update({len(filtered_detections): data})
        for data in objects.values():
            if (target == "ball" and data["cls"] == "black") or (
                target == "evac_point" and data["cls"] == "red"
            ):
                filtered_detections.update({len(filtered_detections): data})

        return filtered_detections

    def get_positions(self, data):
        current_data = {}

        if not data:
            return current_data

        for object in data.values():
            class_name = object["cls"]

            width = float(object["x2"] - object["x1"])
            height = float(object["y2"] - object["y1"])

            centre_x = (object["x1"] + object["x2"]) / 2.0
            centre_y = (object["y1"] + object["y2"]) / 2.0

            if height <= 0:
                self.logger.warning(f"Invalid bounding box height for {class_name}")
                continue

            # Distance estimation from object height (should have less distortion than width)

            if class_name in ["silver", "black"]:
                distance = (self.fy * self.ball_diameter) / height
            elif class_name in ["green", "red"]:
                distance = (self.fy * self.evac_height) / height

            angle = math.degrees(math.atan((centre_x - self.cx) / self.fx))

            self.logger.info(f"Distance to {class_name}: {distance:.2f}m at angle: {angle:.2f}")

            current_data.update({
                len(current_data): {"cls": class_name, "angle": angle, "dist": distance}
            })

        return current_data
