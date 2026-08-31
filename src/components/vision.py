import logger
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO

from src.components.cameras.front_camera import FrontCamera


class Vision:
    conf_threshold = 0.7

    dw = 1536
    dh = 864

    fx = 683.31285
    fy = 683.10689
    cx = 764.89803
    cy = 408.4118

    ball_radius = 0.05
    evac_height = 0.06

    def __init__(self):

        self.camera = FrontCamera()

        self.model = YOLO("src/components/cameras/state.pt")
        self.logger = logger.getLogger("vision")

        self.debug = True

    def _crop_box_to_frame(self, x1, y1, x2, y2):
        return (
            int(x1 + self.start_x),
            int(y1 + self.start_y),
            int(x2 + self.start_x),
            int(y2 + self.start_y),
        )

    def get_all_objects(self):

        all_detections = {}
        counts = {"silver": 0, "black": 0, "green": 0, "red": 0}

        raw_frame, cropped_frame = self.camera.get_frame()
        if raw_frame is None:
            self.logger.info("No frame given!!")
            return

        results = self.model.predict(
            cropped_frame, conf=self.conf_threshold, stream=True, imgsz=640, verbose=False
        )

        for i in results:
            if self.debug:
                annotated_frame = i.plot()
                cv2.imshow("a", annotated_frame)
                cv2.waitKey(1)
                self.out.write(annotated_frame)

            for x1, y1, x2, y2, conf, cls in i.boxes.data.tolist():
                x1, y1, x2, y2 = self._crop_box_to_frame(x1, y1, x2, y2)

                class_name = self.model.names[int(cls)]

                data = {"cls": class_name, "x1": x1, "x2": x2, "y1": y1, "y2": y2, "conf": conf}

                if class_name in counts:
                    counts[class_name] += 1
                all_detections.update({len(all_detections): data})

        return {"detections": all_detections, "counts": counts}

    def filter_objects(self, objects, target):
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

    def get_pos(self, data):
        current_data = {}

        for object in data.values():
            class_name = object["cls"]

            width = float(object["x2"] - object["x1"])
            height = float(object["y2"] - object["y1"])

            centre_x = (object["x1"] + object["x2"]) / 2
            centre_y = (object["y1"] + object["y2"]) / 2

            if class_name in ["silver", "black"]:
                distance = (self.fy * self.evac_height) / height
            elif class_name in ["green", "red"]:
                distance = (self.fy * self.ball_radius) / height

            angle = math.degrees(math.atan((centre_x - self.cx) / self.fx))

            self.logger.info(f"Distance to {class_name}: {distance:.2f}m at angle: {angle:.2f}")

            current_data.update({"cls": class_name, "angle": angle, "dist": distance})

        return current_data
