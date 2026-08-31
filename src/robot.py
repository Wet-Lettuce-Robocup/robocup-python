import logging
import time

from src.components.servo_controller import ServoController


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

    def drive(self, vel: int, angular_vel: int) -> None:
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

    def stop(self) -> None:
        response = self.i2c_controller.handle_write(self.STM_ADDR, self.STOP_REQUEST)
        if not response["success"]:
            self.logger.info(response["message"])

    def drive_dist(self, vel: int, angular_vel: int, drive_time: int) -> None:
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
        dist = self.i2c_controller.read_tof("right")
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
