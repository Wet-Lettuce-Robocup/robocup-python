import logging
import time
from enum import Enum

from components.front_led_controller import LEDController
from components.vision import Vision


class Task(Enum):
    ENTER = 0
    START_SCAN = 1
    SCAN = 2
    TARGET_BALL = 3
    APPROACH_BALL = 4
    LIFT_BALL = 5
    TARGET_EVAC_POINT = 6
    APPROACH_EVAC_POINT = 7
    DUMP_EVAC_POINT = 8
    LOCATE_EXIT = 9
    EXIT = 10


class Rescue:
    DEBUG = True

    def __init__(self, i2c_controller, robot):
        self.logger = logging.getLogger("robot.rescue")

        self.i2c_controller = i2c_controller
        self.robot = robot
        self.vision = Vision(debug=self.DEBUG)
        self.led = LEDController()

        self.current_task = Task.ENTER

        self.target_ball = None
        self.target_evac_point = None

        self.scan_timeout = None

        self.is_targetting_balls = True
        self.green_found = False
        self.red_found = False
        self.exit_found = False

        self.ball_storage = {
            "claw": None,
            "tray_1": None,
            "tray_2": None,
        }

        self.task_started = False

        time.sleep(2)  # To let Ultralytics init

    def reset(self):
        self.current_task = Task.ENTER
        self.task_started = False

    def _transition_to(self, task):
        """Change to a new rescue task."""
        self.logger.info(f"Rescue task: {self.current_task.name} -> {task.name}")
        self.current_task = task
        self.task_started = False

    def set_memory(self, ball_tray_memory):
        if ball_tray_memory is not None:
            self.ball_storage = ball_tray_memory

    def _count_balls(self):
        silver_count = sum(value == "silver" for value in self.ball_storage.values())
        black_count = sum(value == "black" for value in self.ball_storage.values())

        return silver_count, black_count

    def tray_handler(self, action, colour=None):
        """grab, release or dump"""

        current_storage = self.ball_storage
        if action == "release":
            # Claw -> tray_1 if tray_1 & 2 is None
            # Claw -> tray_2 if tray_1 is filled and tray_2 is None
            # Claw -> stays in claw if tray full, shouldn't happen tho
            if current_storage["tray_1"] is None and current_storage["tray_2"] is None:
                self.ball_storage["tray_1"] = current_storage["claw"]
                self.ball_storage["claw"] = None
            elif current_storage["tray_1"] is not None and current_storage["tray_2"] is None:
                self.ball_storage["tray_2"] = current_storage["claw"]
                self.ball_storage["claw"] = None
            elif current_storage["tray_1"] is not None and current_storage["tray_2"] is not None:
                self.logger.info("tray full")

        elif action == "grab":
            # fill the claw
            if current_storage["claw"] is None:
                if colour in ["silver", "black"]:
                    self.ball_storage["claw"] = colour
                else:
                    self.logger.error("Colour is not correct in tray_handler")
            else:
                self.logger.warning("why is there something in the claw man")
        elif action == "dump":
            # tray_1 and tray_2 -> none if both silver
            # tray_1 black and tray_2 empty -> tray_1 empty
            # evac points basically
            if current_storage["tray_1"] == "silver" and current_storage["tray_2"] == "silver":
                self.ball_storage["tray_1"] = None
                self.ball_storage["tray_2"] = None
                self.logger.info("Emptying tray of 2 silver balls")
            elif current_storage["tray_1"] == "black" and current_storage["tray_2"] == None:
                self.ball_storage["tray_1"] = None
                self.logger.info("Emptying tray of 1 black ball")
            else:
                self.logger.warning("Uh oh edge case")

        self.logger.info(
            f"Claw: {self.ball_storage['claw']}, Tray 1: {self.ball_storage['tray_1']}, Tray 2: {self.ball_storage['tray_2']}"
        )

    def locate_targets(self, target):
        all_objects = self.vision.get_all_objects()

        if all_objects is None:
            return None

        counts = all_objects["counts"]
        detections = all_objects["detections"]

        if target == "ball" and counts["silver"] == 0 and counts["black"] == 0:
            return None
        if target == "evac_point" and counts["green"] == 0 and counts["red"] == 0:
            return None

        target_objects = self.vision.filter_objects(detections, target)

        if len(target_objects) == 0:
            return None

        target_positions = self.vision.get_positions(target_objects)

        return target_positions

    def scan_for_balls(self):
        silver_found, black_found = self._count_balls()

        if time.monotonic() > self.scan_timeout:
            self.logger.info("Scan timeout reached, moving to find more balls")
            self.robot.drive_PID(300, 100)
            time.sleep(1.5)
            self.robot.stop_moving()
            return []

        elif silver_found == 2 and black_found == 1:
            self.logger.info(f"All balls rescued: Silver = {silver_found}, Black = {black_found}")
            self.is_targetting_balls = False
            return []

        elif silver_found < 2:
            target_colour = "silver"

        elif black_found < 1:
            target_colour = "black"

        ball_positions = self.locate_targets("ball")

        if not ball_positions:
            return []

        final_positions = []
        for i in ball_positions.values():
            if i["cls"] == target_colour:
                final_positions.append(i)

        if not final_positions:
            self.logger.info(f"No {target_colour} ball detected")
            return []

        final_positions.sort(key=lambda x: x["dist"])
        return final_positions

    def scan_for_evac_points(self):
        evac_positions = self.locate_targets("evac_point")

        if not evac_positions:
            return []

        self.logger.info(f"Evac positions: {evac_positions}")
        # Separate green and red detections
        green = [position for position in evac_positions.values() if position["cls"] == "green"]
        red = [position for position in evac_positions.values() if position["cls"] == "red"]

        target = self.get_target_evac_point()

        if target == "green" and green:
            return green
        if target == "red" and red:
            return red

        return []

    def get_target_evac_point(self):
        tray_1 = self.ball_storage["tray_1"]
        tray_2 = self.ball_storage["tray_2"]

        if tray_1 == "silver" and tray_2 == "silver":
            return "green"
        elif tray_1 == "red":
            return "red"
        return ""

    def rotate_to_target(self, angle):
        if abs(angle) < 5:
            self.logger.info(f"Target angle {angle:.1f}°, no rotation required")
            return

        self.logger.info(f"Rotating {angle:.1f}° towards target")

        self.robot.spin_enc(angle, 200)
        time.sleep(5)

    def move_to_target(self, distance):
        if distance <= 0:
            self.logger.warning(f"Invalid target distance: {distance}")
            return

        self.logger.info(f"Driving {distance:.2f}m towards target")
        self.robot.drive_dist_enc(distance * 1000, 300)
        time.sleep(5)

    def enter_rescue(self):
        self.logger.info("Entering rescue zone")

        self.robot.stop_moving()

        self.robot.claw("grab")
        self.robot.lift("up")
        self.robot.tray("reset")

        self.led.set_brightness(0)

        dist = self.robot.get_front_distance()  # it's in mm btw

        if 200 < dist < 1000:
            self.robot.drive_dist_enc(dist / 2)
        else:
            self.logger.warning("Front dist not valid, driving 400mm anyway")
            self.robot.drive_dist_enc(400)
        time.sleep(4)

        left_dist = self.robot.get_side_distance()
        if 0 < left_dist < 300:
            self.robot.spin_enc(45, 500)
            time.sleep(2)
            self.robot.drive_dist_enc(300)
            time.sleep(3)

    def grab_ball(self):
        if self.target_ball is None:
            self.logger.warning("grab_ball called without a target")
            self._transition_to(Task.SCAN)
            return

        self.robot.lift("down")

        colour = self.target_ball["cls"]
        self.logger.info(f"Grabbing {colour} ball")

        self.robot.drive_dist_enc(50, velocity=250)
        time.sleep(4)

        self.robot.claw("grab")

        # Assume the ball was successfully grabbed.
        # # The claw TOF can be used here later if required.
        claw_distance = self.robot.get_claw_distance()
        self.logger.info(f"Claw distance after grab: {claw_distance}mm")

        self.tray_handler("grab", colour)

        # Reverse away from the ball.
        self.robot.drive_dist_enc(-150, 400)
        time.sleep(3)

        # Lift the ball.
        self.robot.lift("up")
        self.target_ball = None

    def dump_balls(self):
        if self.target_evac_point is None:
            self.logger.warning("dump_balls called without evacuation point")
            return

        colour = self.target_evac_point["cls"]
        self.logger.info(f"Dumping balls at {colour} evacuation point")

        # Back away from the evacuation point.
        self.robot.drive_dist_enc(-100, 500)
        time.sleep(4)

        # Turn around.
        self.robot.spin_enc(180)
        time.sleep(10)

        # Move backwards/towards the drop area.
        self.robot.drive_dist_enc(-150, 400)
        time.sleep(4)

        self.robot.tray("release")
        time.sleep(3)
        self.robot.tray("reset")

        self.robot.drive_dist_enc(300, 400)

        if (
            self.ball_storage["tray_1"] == self.ball_storage["tray_2"]
            or self.ball_storage["claw"] is not None
        ):
            # Released two balls
            self.robot.claw("release")
            time.sleep(1)
            self.robot.claw("grab")

        self.tray_handler("release")

        self.target_evac_point = None

        if colour == "green":
            self.green_found = True
        elif colour == "red":
            self.red_found = True

        self.logger.info(f"Finished dumping at {colour} evacuation point")

    def left_wall_follow(self, target_distance=150):
        """
        Basic left-wall following.
        Distances returned by Robot are in mm.
        Returns: ("exit", angle) ("right", angle) ("wall", angle) ("error", angle)
        """

        left_dist = self.robot.get_side_distance()
        front_dist = self.robot.get_front_distance()

        if left_dist is None:
            self.logger.error("Left tof not working")
            return "error", 0
        if front_dist is None:
            self.logger.error("Front tof not working")
            return "error", 0

        if front_dist < 0 or left_dist < 0:
            return ("error", 0)

        # Wall directly ahead.
        if front_dist < 200:
            return "right", 0

        # No wall on left means we may have reached the exit.
        if left_dist >= 400:
            return "exit", 0

        error = target_distance - left_dist

        # Convert wall distance error into a small turning angle.
        angle = error * 0.03
        angle = max(-1.5, min(1.5, angle))
        return "wall", angle

    def locate_exit(self):
        """Follow the left wall until the rescue exit is found."""

        self.logger.info("Locating rescue exit")

        while True:
            status, angle = self.left_wall_follow()

            if status == "error":
                self.logger.warning("Invalid TOF data during wall following")
                self.robot.stop_moving()
                time.sleep(0.1)
                continue

            if status == "exit":
                self.logger.info("Rescue exit detected")
                self.robot.stop_moving()
                self.exit_found = True
                return

            if status == "right":
                self.logger.info("Wall ahead, turning right")
                self.robot.stop_moving()
                self.robot.spin_enc(90)
                time.sleep(5)
                continue

            if status == "wall":
                # Small movement while correcting against wall.
                self.robot.drive_PID(vel=320, angular_vel=int(angle * 60))
                time.sleep(0.1)

    def get_debug_frame(self):
        frame = self.vision.get_camera_frame()
        if frame is not None:
            return frame
        return None

    def tick_rescue(self):
        if self.current_task == Task.ENTER:
            if self.task_started:
                return

            self.task_started = True

            self.target_ball = None
            self.target_evac_point = None

            self.scan_timeout = None

            self.is_targetting_balls = True
            self.green_found = False
            self.red_found = False
            self.exit_found = False

            self.enter_rescue()
            self._transition_to(Task.START_SCAN)

        elif self.current_task == Task.START_SCAN:
            # set up scanning for a ball or evac point
            if self.task_started:
                return

            self.task_started = True
            self.led.set_brightness(30)
            self._transition_to(Task.SCAN)

        elif self.current_task == Task.SCAN:
            # scan for ball or evac point
            first_run = False
            if not self.task_started:
                first_run = True
                # Timeout of 60 seconds if no black or silver
                self.scan_timeout = time.monotonic() + 60

            self.task_started = True

            if self.is_targetting_balls:
                positions = self.scan_for_balls()
                if not positions:
                    if first_run:
                        self.robot.drive_PID(0, 250)
                    return

                self.robot.stop_moving()

                self.target_ball = positions[0]

                self.logger.info(
                    f"Targeting {self.target_ball['cls']} ball: "
                    f"{self.target_ball['dist']:.2f}m, "
                    f"{self.target_ball['angle']:.1f}°"
                )
                self._transition_to(Task.TARGET_BALL)

            else:
                # All balls collected.
                positions = self.scan_for_evac_points()

                if not positions:
                    self.logger.info("No evacuation point found")

                    if first_run:
                        self.robot.drive_PID(0, 250)
                    return

                self.robot.stop_moving()

                self.target_evac_point = positions[0]

                self.logger.info(f"Targeting {self.target_evac_point['cls']} evacuation point")
                self._transition_to(Task.TARGET_EVAC_POINT)

        elif self.current_task == Task.TARGET_BALL:
            # turn to face a ball
            if self.target_ball is None:
                self._transition_to(Task.SCAN)
                return
            if self.task_started:
                return

            self.task_started = True

            self.logger.info(f"Rotating to {self.target_ball['cls']} ball")

            self.robot.claw("release")
            self.tray_handler("release")

            self.rotate_to_target(self.target_ball["angle"])

            self._transition_to(Task.APPROACH_BALL)

        elif self.current_task == Task.APPROACH_BALL:
            # move to approach a ball
            if self.target_ball is None:
                self._transition_to(Task.SCAN)
                return
            if self.task_started:
                return

            self.task_started = True

            # Stop short of ball and double check distance.
            distance = self.target_ball["dist"]
            approach_distance = max(0.0, distance - 0.15)
            self.logger.info(f"Approaching ball: {approach_distance:.2f}m")
            if approach_distance > 0:
                self.robot.drive_dist_enc(approach_distance * 1000, 350)
                time.sleep(4)
            self._transition_to(Task.LIFT_BALL)

        elif self.current_task == Task.LIFT_BALL:
            # lift up ball and check if legit
            if self.task_started:
                return

            self.task_started = True

            # Re-check the ball position after approaching.
            positions = self.scan_for_balls()

            if not positions:
                self.logger.warning("Ball lost during approach")
                self.target_ball = None
                self._transition_to(Task.SCAN)
                return

            self.target_ball = positions[0]

            # Rotate again using the updated position.
            self.rotate_to_target(self.target_ball["angle"])

            # Move close enough for the claw.
            distance = self.target_ball["dist"]
            drive_dist = max(0, distance - 0.08)

            if distance > 0.10:
                self.robot.drive_dist_enc(drive_dist * 1000, 300)
                time.sleep(3)

            self.grab_ball()

            self._transition_to(Task.SCAN)

        elif self.current_task == Task.TARGET_EVAC_POINT:
            # turn to face evac point
            if self.target_evac_point is None:
                self._transition_to(Task.SCAN)
                return
            if self.task_started:
                return

            self.task_started = True

            self.logger.info(f"Rotating towards {self.target_evac_point['cls']} evacuation point")
            self.rotate_to_target(self.target_evac_point["angle"])

            self._transition_to(Task.APPROACH_EVAC_POINT)

        elif self.current_task == Task.APPROACH_EVAC_POINT:
            # move to approach evac point
            if self.target_evac_point is None:
                self._transition_to(Task.SCAN)
                return
            if self.task_started:
                return

            self.task_started = True

            distance = self.target_evac_point["dist"]

            # Stop slightly short of the evacuation point.
            approach_distance = max(0.0, distance - 0.15)

            self.logger.info(f"Approaching evacuation point: {approach_distance:.2f}m")

            if approach_distance > 0:
                self.robot.drive_dist_enc(approach_distance * 1000)
                time.sleep(5)

            self._transition_to(Task.DUMP_EVAC_POINT)

        elif self.current_task == Task.DUMP_EVAC_POINT:
            # turn and drop balls off at evac point

            if self.task_started:
                return

            self.task_started = True

            self.dump_balls()

            if self.red_found:
                self._transition_to(Task.LOCATE_EXIT)
            else:
                self._transition_to(Task.SCAN)

        elif self.current_task == Task.LOCATE_EXIT:
            # loop to find exit
            if not self.task_started:
                self.task_started = True

                self.led.set_brightness(0)

                self.robot.drive_dist_enc(200)
                time.sleep(3)

            self.locate_exit()

            self._transition_to(Task.EXIT)

        elif self.current_task == Task.EXIT:
            # end rescue and start line follow
            if self.task_started:
                return

            self.task_started = True

            self.robot.stop_moving()
            self.led.set_brightness(0)

            self.logger.info("Rescue complete")

        else:
            self.logger.error(f"Unknown rescue task: {self.current_task}")

    def is_finished(self):
        """Return True when rescue has completed."""
        return self.current_task == Task.EXIT

    def exit(self):
        self.vision.close()
        self.led.set_brightness(0)
        self.robot.claw("grab")
        self.robot.lift("up")
        self.robot.tray("reset")
