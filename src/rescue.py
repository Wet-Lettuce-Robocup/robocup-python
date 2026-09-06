import logging
import time

from src.components.vision import Vision
from src.robot import Robot


class Rescue:
    def __init__(self, i2c_controller):
        self.logger = logging.getLogger("rescue")

        self.i2c_controller = i2c_controller
        self.robot = Robot(self.i2c_controller)
        self.vision = Vision()

        self.ball_positions = {}
        self.evac_positions = {}

    def tick_rescue(self):
        pass

    def locate_targets(self, target):
        all_objects = self.vision.get_all_objects()

        counts = all_objects["counts"]
        detections = all_objects["detections"]

        if target == "ball" and counts["silver"] == 0 and counts["black"] == 0:
            return
        if target == "evac_point" and counts["green"] == 0 and counts["red"] == 0:
            return

        target_objects = self.vision.filter_objects(detections, target)

        if len(target_objects) == 0:
            return

        target_positions = self.vision.get_positions(target_objects)

        return target_positions

    def target_ball(self):
        self.ball_positions = self.locate_targets("ball")

    def target_evac_points(self):
        self.evac_positions = self.locate_targets("evac_point")
