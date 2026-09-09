import logging
import math
import time

from gpiozero import Button

from components.servo_controller import ServoController


class Robot:
    STM_ADDR = 0x67
    DRIVE_REQUEST = 0x01
    STOP_REQUEST = 0x02
    DRIVE_TIME_REQUEST = 0x04
    ENCODER_REQUEST = 0x82
    ENCODER_LEN = 16

    def __init__(self, i2c_controller) -> None:

        self.logger = logging.getLogger("robot")

        self.i2c_controller = i2c_controller

        self.servo_grab = ServoController(self.i2c_controller, servo_id=0, gpio_pin=4)
        self.servo_lift = ServoController(self.i2c_controller, servo_id=1, gpio_pin=0)
        self.servo_tray_release = ServoController(self.i2c_controller, servo_id=2, gpio_pin=1)

        self.limit_switch = Button(27, pull_up=False)

        self.limit_switch_triggered = False

        self.limit_switch.when_pressed = self._limit_switch_was_pressed

    def drive(self, vel: int = 50, angular_vel: int = 0) -> None:
        vel = int(vel)
        angular_vel = int(angular_vel)

        data = [
            (vel >> 24) & 0xFF,
            (vel >> 16) & 0xFF,
            (vel >> 8) & 0xFF,
            vel & 0xFF,
            0,
            0,
            0,
            0,
            (angular_vel >> 24) & 0xFF,
            (angular_vel >> 16) & 0xFF,
            (angular_vel >> 8) & 0xFF,
            angular_vel & 0xFF,
        ]

        response = self.i2c_controller.handle_write(self.STM_ADDR, self.DRIVE_REQUEST, data)
        if not response["success"]:
            self.logger.info(response["message"])

    def stop_moving(self) -> None:
        response = self.i2c_controller.handle_write(self.STM_ADDR, self.STOP_REQUEST)
        if not response["success"]:
            self.logger.info(response["message"])

    def spin(self, angle, velocity: int = 50):
        self.drive_dist(0, angle, velocity)

    def drive_dist(self, distance, angle=0, velocity: int = 100) -> None:
        distance *= 500  # To tune
        angle *= 1.2

        linear_time = abs(distance) / abs(velocity) if velocity != 0 and distance != 0 else 0.0
        angular_time = abs(angle) / abs(velocity) if velocity != 0 and angle != 0 else 0.0
        time_required = max(linear_time, angular_time)

        if time_required <= 0.0:
            self.logger.warning("Ignoring drive called with zero distance and angle")
            return

        linear_vel = math.copysign(velocity, distance) if distance != 0 else 0.0
        angular_vel = math.copysign(velocity, angle) if angle != 0 else 0.0
        vel = int(linear_vel)
        angular_vel = int(angular_vel)
        drive_time = int(time_required)

        data = [
            (vel >> 24) & 0xFF,
            (vel >> 16) & 0xFF,
            (vel >> 8) & 0xFF,
            vel & 0xFF,
            0,
            0,
            0,
            0,
            (angular_vel >> 24) & 0xFF,
            (angular_vel >> 16) & 0xFF,
            (angular_vel >> 8) & 0xFF,
            angular_vel & 0xFF,
            (drive_time >> 24) & 0xFF,
            (drive_time >> 16) & 0xFF,
            (drive_time >> 8) & 0xFF,
            drive_time & 0xFF,
        ]

        response = self.i2c_controller.handle_write(self.STM_ADDR, self.DRIVE_TIME_REQUEST, data)
        if not response["success"]:
            self.logger.info(response["message"])

        time.sleep(drive_time / 1000)

    def get_encoders(self) -> tuple[int, int, int, int]:
        response = self.i2c_controller.handle_read(
            self.STM_ADDR, self.ENCODER_REQUEST, self.ENCODER_LEN
        )
        if not response["success"]:
            self.logger.info(response["message"])
            return

        data = response["data"]

        fl = int.from_bytes(data[0:4], signed=True)
        fr = int.from_bytes(data[4:8], signed=True)
        bl = int.from_bytes(data[8:12], signed=True)
        br = int.from_bytes(data[12:16], signed=True)

        encoders = (fl, fr, bl, br)
        return encoders

    def get_side_distance(self) -> int:
        dist = self.i2c_controller.read_tof("side")
        if dist > 0:
            return dist
        return -1

    def get_front_distance(self) -> int:
        dist = self.i2c_controller.read_tof("front")
        if dist > 0:
            return dist
        return -1

    def get_claw_distance(self) -> int:
        dist = self.i2c_controller.read_tof("claw")
        if dist > 0:
            return dist
        return -1

    def claw(self, action):
        """Action: "grab" or "release"."""
        if action == "grab":
            self.servo_grab.set_angle(54)
            time.sleep(0.5)
        elif action == "release":
            self.servo_grab.set_angle(29)
            time.sleep(0.5)

    def lift(self, action):
        """Action: "up" or "down"."""
        if action == "up":
            self.servo_lift.set_angle(155)
            time.sleep(0.5)
        elif action == "down":
            self.servo_lift.set_angle(23)
            time.sleep(0.5)

    def tray(self, action):
        """Action: "release" or "reset"."""
        if action == "release":
            self.servo_tray_release.set_angle(132)
            time.sleep(0.5)
        elif action == "reset":
            self.servo_tray_release.set_angle(46)
            time.sleep(0.5)

    def limit_switch_pressed(self):
        """Returns True if limit switch is pressed in that moment."""
        return self.limit_switch.is_pressed

    def _limit_switch_was_pressed(self):
        """Check if limit switch was pressed after last reset (for rescue, may not use)"""
        self.limit_switch_triggered = True

    def reset_limit_switch(self):
        self.limit_switch_triggered = False
