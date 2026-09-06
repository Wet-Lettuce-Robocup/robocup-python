import cv2
import logging
import math
import numpy as np
import time

from src.components.cameras.front_camera import FrontCamera


class Follow:
    VELOCITY = 50

    def __init__(self, i2c_controller, robot):
        self.logger = logging.getLogger("line_follow")

        self.i2c_controller = i2c_controller
        self.robot = robot
        self.camera = FrontCamera()

        self.frame = None

        self.lastError = 0
        self.pastErrors = 0
        self.distance = 0
        self.loops = 0
        self.startTime = time.monotonic()

        self.follow_active = True

    def follow(self) -> float:
        """2025 line follow code"""
        self.frame = self.camera.get_frame()

        # Check if frame is valid
        if self.frame is None:
            self.logger.warning("get_frame returned None")
            return 0.0

        self.img = np.copy(self.frame)
        self.green = self.getGreen()

        self.roi = self.frame[0:360, 0 : self.frameWidth]
        self.line = cv2.inRange(self.roi, (0, 0, 0), (60, 60, 60))  # Black threshold

        greenROI = self.green[0 : len(self.roi), 0 : self.frameWidth]
        green_weight = 2
        weighted_combined_mask = cv2.addWeighted(greenROI, green_weight, self.line, 1, 0)
        combined_mask = (weighted_combined_mask > 0).astype(np.uint8) * 255

        self.line = cv2.bitwise_and(self.roi, self.roi, mask=combined_mask)

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

        greenContours = cv2.findContours(greenROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(greenContours) > 0:
            largestContourGreen = max(greenContours, key=cv2.contourArea)
            mGreen = cv2.moments(largestContourGreen)
            if mGreen["m00"] != 0:
                self.xPosGreen = int(mGreen["m10"] / mGreen["m00"])
                self.yPosGreen = int(mGreen["m01"] / mGreen["m00"])
            else:
                self.xPosGreen = 0
                self.yPosGreen = 0
        else:
            largestContourGreen = None

        # Average green and line COM
        if largestContourGreen is not None and cv2.contourArea(largestContourGreen) > 1000:
            self.xPos = int((self.xPos + self.xPosGreen) / 2)
            self.yPos = int((self.yPos + self.yPosGreen) / 2)
        else:
            self.xPosGreen = 0
            self.yPosGreen = 0

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
        self.frame = self.camera.get_frame()

        if self.frame is None:
            return False

        line = cv2.inRange(self.img, (0, 0, 0), (45, 45, 45))
        return cv2.countNonZero(line) > 5000

    def water_tower(self):
        self.robot.spin(90)
        self.robot.drive_dist(30, -10, 40)  # to tune
        self.robot.drive(20)
        while not self.lineInFrame():
            pass
        self.robot.stop_moving()

    def main(self):
        while self.follow_active:
            if self.robot.limit_switch_pressed():
                self.robot.stop_moving()
                self.water_tower()
            else:
                self.loop()

    def loop(self):
        self.loops += 1
        self.lastDistance = self.distance

        # error calc
        angle = self.follow()

        # angle = pid.calcTurnRate(angle, 1.4, 0, 0, self.lastError, self.pastErrors)

        self.robot.drive(self.VELOCITY, angle)
