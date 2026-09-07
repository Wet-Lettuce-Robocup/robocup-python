import logging
import math
import time
from enum import Enum

import cv2
import numpy as np

from src.components.cameras.front_camera import FrontCamera


class Task(Enum):
    INIT = 1
    FOLLOW = 2
    TOWER = 3
    RESCUE = 4


class Follow:
    VELOCITY = 50
    MIN_RED_AREA = 500.0

    def __init__(self, i2c_controller, robot):
        self.logger = logging.getLogger("line_follow")

        self.i2c_controller = i2c_controller
        self.robot = robot
        self.camera = FrontCamera()

        self.raw_frame = None
        self.cropped_frame = None

        self.startTime = time.monotonic()

        self.follow_status = Task.INIT
        self.task_started = False

    def _transition_to(self, task):
        self.logger.info(f"Task: {self.follow_statuss.name} -> {task.name}")

        self.follow_status = task
        self.task_started = False

    def follow(self, frame) -> float:
        """2025 line follow code"""

        # Check if frame is valid
        if frame is None:
            self.logger.warning("Not a valid frame")
            return 0.0

        self.img = np.copy(frame)
        # self.green = self.getGreen()

        self.roi = self.frame[0:360, 0 : self.frameWidth]
        self.line = cv2.inRange(self.roi, (0, 0, 0), (60, 60, 60))  # Black threshold

        # greenROI = self.green[0 : len(self.roi), 0 : self.frameWidth]
        green_weight = 2
        # weighted_combined_mask = cv2.addWeighted(greenROI, green_weight, self.line, 1, 0)
        # combined_mask = (weighted_combined_mask > 0).astype(np.uint8) * 255

        # self.line = cv2.bitwise_and(self.roi, self.roi, mask=combined_mask)

        if len(self.line.shape) == 3:
            self.line = cv2.cvtColor(self.line, cv2.COLOR_BGR2GRAY)

        self.line = (self.line > 0).astype(np.uint8) * 255
        lineContours, _ = cv2.findContours(self.line, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        linePos = 0

        if len(self.line) <= 100 or len(lineContours) == 0:
            return linePos

        largestContourLine = max(lineContours, key=cv2.contourArea)
        m = cv2.moments(largestContourLine)
        if m["m00"] == 0:
            return linePos
        self.xPos = int(m["m10"] / m["m00"])
        self.yPos = int(m["m01"] / m["m00"])

        # greenContours = cv2.findContours(greenROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # if len(greenContours) > 0:
        #     largestContourGreen = max(greenContours, key=cv2.contourArea)
        #     mGreen = cv2.moments(largestContourGreen)
        #     if mGreen["m00"] != 0:
        #         self.xPosGreen = int(mGreen["m10"] / mGreen["m00"])
        #         self.yPosGreen = int(mGreen["m01"] / mGreen["m00"])
        #     else:
        #         self.xPosGreen = 0
        #         self.yPosGreen = 0
        # else:
        #     largestContourGreen = None

        # Average green and line COM
        # if largestContourGreen is not None and cv2.contourArea(largestContourGreen) > 1000:
        #     self.xPos = int((self.xPos + self.xPosGreen) / 2)
        #     self.yPos = int((self.yPos + self.yPosGreen) / 2)
        # else:
        #     self.xPosGreen = 0
        #     self.yPosGreen = 0

        # REMOVE THIS LATER ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if len(lineContours) > 0:
        #     cv2.drawContours(self.img, lineContours, -1, (0, 0, 255), 2)
        #     cv2.drawContours(self.img, [largestContourLine], -1, (0, 255, 0), 3)
        #     cv2.circle(self.img, (self.xPos, self.yPos), 5, (255, 0, 0), -1)
        #     if len(greenContours) > 0:
        #         cv2.drawContours(self.img, greenContours, -1, (255, 0, 255), 3)
        #     else:
        #         print("No green ROI detected")
        #     center_x, center_y = 240, 360
        #     cv2.line(self.img, (center_x, center_y), (self.xPos, self.yPos), (0, 255, 255), 2)
        # cv2.imshow("Line Following - Black Line Contours", self.img)
        # cv2.waitKey(1)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if self.yPos > 300:  # if line is behind robot (adjusted for 360 height)
            if self.xPos > 240:  # if line is to the right
                linePos = 100
            elif self.xPos < 240:  # if line is to the left
                linePos = -100
            else:
                self.logger.info("line is really cooked its fully behind")
                linePos = 0
        elif self.yPos == 360:
            linePos = 0
        else:
            linePos = math.degrees(math.atan((self.xPos - 240) / (360 - self.yPos)))

        # print(f"linepos: {linePos:.0f}")
        return linePos

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
            frame = self.cropped_frame

            if self.red_detected(frame):
                self._transition_to(Task.RESCUE)
                return

            # error calc
            angle = self.follow(frame)

            # angle = pid.calcTurnRate(angle, 1.4, 0, 0, self.lastError, self.pastErrors)

            self.robot.drive(self.VELOCITY, angle)

        elif self.follow_status == Task.INIT:
            if not self.task_started:
                self._transition_to(Task.FOLLOW)
                self.task_started = True

    def is_finished(self):
        return self.follow_status == Task.RESCUE
