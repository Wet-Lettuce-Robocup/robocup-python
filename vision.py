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

    def get_all_objects(self):

        all_detections = {}
        counts = {"silver": 0, "black": 0, "green": 0, "red": 0}

        raw_frame, cropped_frame = self.camera.get_frame()
        if raw_frame is None:
            print("No frame given!!")
            return

        results = self.model.predict(
            cropped_frame, conf=self.conf, stream=True, imgsz=self.imgsz, verbose=False
        )

        for i in results:
            if self.debug:
                annotated_frame = i.plot()
                cv2.imshow("a", annotated_frame)
                cv2.waitKey(1)
                self.out.write(annotated_frame)

            for x1, y1, x2, y2, conf, cls in i.boxes.data.tolist():
                # self.get_logger().info(str(i.boxes))

                x1, y1, x2, y2 = self.crop_box_to_frame(x1, y1, x2, y2)

                class_name = self.model.names[int(cls)]
                # self.get_logger().info(f'{class_name} detected')

                data = {"cls": class_name, "x1": x1, "x2": x2, "y1": y1, "y2": y2, "conf": conf}

                if class_name in counts:
                    counts[class_name] += 1
                all_detections.update({len(all_detections): data})

        return {"detections": all_detections, "counts": counts}

    def filter_objects(self, objects):

        # annotated_image = dets[0].plot()

        # ---------------------------------------
        # 1) First try YOLO detections
        # ---------------------------------------
        for box in dets[0].boxes:
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.pt.names[class_id]

            x1, y1, x2, y2 = xyxy

            if confidence > 0.6:
                all_dets.append([class_name, x1, y1, x2, y2, confidence])

                # Optional drawing for debug:
                # cv2.circle(annotated_image, (x, y), r, (0, 255, 0), 2)
                # cv2.circle(annotated_image, (x, y), 2, (0, 0, 255), 3)
                # cv2.putText(
                #     annotated_image,
                #     colour,
                #     (x - r, max(20, y - r - 10)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.55,
                #     (0, 255, 255),
                #     2,
                #     cv2.LINE_AA,
                # )

        final_data = self.get_nums(all_dets)

        # Optional debug window
        # cv2.imshow("Ball colour detection", annotated_image)
        # key = cv2.waitKey(1) & 0xFF
        # if key == 27 or key == ord("q"):
        #     return

        return final_data

    def get_nums(self, data):
        current_data = []

        if len(data) == 0:
            return

        for ball in data:
            class_name = ball[0]
            x1 = ball[1]
            y1 = ball[2]
            x2 = ball[3]
            y2 = ball[4]

            width = x2 - x1
            height = y2 - y1

            centre_x = (x1 + x2) / 2

            average_dimension = (width + height) / 2
            distance = (self.fx * self.ball_radius) / average_dimension
            angle = math.atan((centre_x - self.cx) / self.fx)

            if self.request_count % self.log_interval == 0:
                print(f"Distance to {class_name}: {distance:.2f}m at angle: {angle:.2f}")
            current_data.append([class_name, angle, distance])

        return current_data
