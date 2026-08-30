#!/usr/bin/env python3

from smbus2 import smbus2
import adafruit_vl53l1x
import board
from gpiozero import OutputDevice
import time
from servos import Servo


class Robot:
    STM_ADDR = 0x67
    DRIVE_REQUEST = 0x01
    STOP_REQUEST = 0x02
    DRIVE_TIME_REQUEST = 0x04
    ENCODER_REQUEST = 0x82
    ENCODER_LEN = 16

    def __init__(self) -> None:
        self.bus = smbus2.SMBus(1)

        self.servo_grab = Servo(0, 4)
        self.servo_lift = Servo(1, 0)
        self.servo_tray_release = Servo(2, 1)

        i2c = board.I2C()

        print("Initializing ToF sensors...")

        front_xshut = OutputDevice(7, active_high=True)
        side_xshut = OutputDevice(19, active_high=True)

        front_xshut.off()
        side_xshut.off()
        time.sleep(0.5)

        front_xshut.on()
        time.sleep(0.5)

        self.fronttof = adafruit_vl53l1x.VL53L1X(i2c)
        print("Front ToF sensor initialized.")
        self.fronttof.set_address(0x30)

        side_xshut.on()
        time.sleep(0.05)

        self.sidevl53 = adafruit_vl53l1x.VL53L1X(i2c)
        print("Side ToF sensor initialized.")

        self.fronttof.start_ranging()
        self.sidevl53.start_ranging()

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

        self.bus.write_i2c_block_data(self.STM_ADDR, self.DRIVE_REQUEST, data)

    def stop(self) -> None:
        self.bus.write_byte(self.STM_ADDR, self.STOP_REQUEST)

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

        self.bus.write_i2c_block_data(self.STM_ADDR, self.DRIVE_TIME_REQUEST, data)

        time.sleep(drive_time / 1000)

    def get_encoders(self) -> tuple[int, int, int, int]:
        data = self.bus.read_i2c_block_data(self.STM_ADDR, self.ENCODER_REQUEST, self.ENCODER_LEN)

        fl = int.from_bytes(data[0:4], signed=True)
        fr = int.from_bytes(data[4:8], signed=True)
        bl = int.from_bytes(data[8:12], signed=True)
        br = int.from_bytes(data[12:16], signed=True)

        encoders = (fl, fr, bl, br)
        return encoders

    def get_side_distance(self) -> int:
        if self.sidevl53.data_ready:
            return self.sidevl53.distance
        return -1

    def get_front_distance(self) -> int:
        if self.fronttof.data_ready:
            return self.fronttof.distance
        return -1
