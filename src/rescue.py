from enum import Enum
import logging
import time

from src.components.vision import Vision
from src.components.front_led_controller import LEDController
from src.robot import Robot


class Tasks(Enum):
    ENTER = 0
    START_SCAN = 1
    SCAN = 2
    STOP_SCAN = 3
    TARGET_BALL = 4
    APPROACH_BALL = 5
    LIFT_BALL = 6
    TARGET_EVAC_POINT = 7
    APPROACH_EVAC_POINT = 8
    DUMP_EVAC_POINT = 9
    LOCATE_EXIT = 10
    EXIT = 11


class Rescue:
    def __init__(self, i2c_controller):
        self.logger = logging.getLogger("rescue")

        self.i2c_controller = i2c_controller
        self.robot = Robot(self.i2c_controller)
        self.vision = Vision()
        self.led = LEDController()

        self.ball_positions = {}
        self.evac_positions = {}

        self.current_task = Tasks.ENTER

        self.target_attempts = 0

        self.green_found = False
        self.red_found = False
        self.exit_found = False

        self.ball_storage = {
            "claw": None,
            "tray_1": None,
            "tray_2": None,
        }

        time.sleep(2)  # To let Ultralytics init

    def count_balls(self):
        silver_count = sum(value == "silver" for value in self.ball_storage.values())
        black_count = sum(value == "black" for value in self.ball_storage.values())

        return silver_count, black_count

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

    def target_ball(self, colour):
        self.ball_positions = self.locate_targets("ball")

    def target_evac_point(self, colour):
        self.evac_positions = self.locate_targets("evac_point")

    def rotate_to_target(self, angle):
        pass

    def move_to_target(self, dist):
        pass

    def enter_rescue(self):
        self.robot.stop_moving()
        self.robot.claw("grab")
        self.robot.lift("up")
        self.robot.tray("reset")

        self.led.set_brightness(0)

        dist = self.robot.get_front_distance()  # it's in mm btw

        if dist > 0.2 and dist < 1:
            self.robot.drive_dist(dist / 2)
        else:
            self.robot.drive_dist(0.4)

        left_dist = self.robot.get_side_distance()
        if left_dist > 0 and left_dist < 0.3:
            self.robot.spin(90)
            self.robot.drive_dist(0.3)

    def tick_rescue(self):
        if self.current_task == Tasks.ENTER:
            self.enter_rescue()
            self.current_task = Tasks.START_SCAN

        elif self.current_task == Tasks.START_SCAN:
            # set up scanning for a ball or evac point
            self.led.set_brightness(30)
            self.current_task = Tasks.SCAN

        elif self.current_task == Tasks.SCAN:
            # scan for ball or evac point
            silver_found, black_found = self.count_balls()
            if silver_found < 2:
                self.target_ball("silver")
                self.target_attempts += 1
            elif black_found < 1:
                self.target_ball("black")
            elif not self.green_found:
                self.target_evac_point("green")
            elif not self.red_found:
                self.target_evac_point("red")

        elif self.current_task == Tasks.STOP_SCAN:
            # turn off scanning
            pass

        elif self.current_task == Tasks.TARGET_BALL:
            # turn to face a ball
            pass

        elif self.current_task == Tasks.APPROACH_BALL:
            # move to approach a ball
            pass

        elif self.current_task == Tasks.LIFT_BALL:
            # lift up ball and check if legit
            pass

        elif self.current_task == Tasks.TARGET_EVAC_POINT:
            # turn to face evac point
            pass

        elif self.current_task == Tasks.APPROACH_EVAC_POINT:
            # move to approach evac point
            pass

        elif self.current_task == Tasks.DUMP_EVAC_POINT:
            # turn and drop balls off at evac point
            pass

        elif self.current_task == Tasks.LOCATE_EXIT:
            # loop to find exit
            pass

        elif self.current_task == Tasks.EXIT:
            # end rescue and start line follow
            pass
