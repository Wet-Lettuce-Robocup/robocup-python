import time

from src.components.vision import Vision
from src.robot import Robot


class Rescue:
    def __init__(self):
        self.robot = Robot()
        self.vision = Vision()

    def claw(self, action):
        if action == "grab":
            self.robot.servo_grab.move_angle(57)
            time.sleep(0.5)
        elif action == "release":
            self.robot.servo_grab.move_angle(29)
            time.sleep(0.5)

    def lift(self, action):
        if action == "up":
            self.robot.servo_lift.move_angle(155)
            time.sleep(0.5)
        elif action == "down":
            self.robot.servo_lift.move_angle(23)
            time.sleep(0.5)

    def tray(self, action):
        if action == "release":
            self.robot.servo_tray_release.move_angle(132)
            time.sleep(0.5)
        elif action == "reset":
            self.robot.servo_tray_release.move_angle(46)
            time.sleep(0.5)
