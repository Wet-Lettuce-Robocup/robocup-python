import adafruit_vl53l1x
import board
import busio

import logging
import threading
from smbus2 import SMBus

from gpiozero import OutputDevice


class I2CBusController:
    CLAW_TOF_ADDR = 0x30
    SIDE_TOF_ADDR = 0x31
    FRONT_TOF_ADDR = 0x32
    STM_ADDR = 0x67

    STATE_CMD = 0x80
    ENCODER_REQUEST = 0x82
    ULTRASONIC_CMD = 0x83
    TEMP_CMD = 0x84
    MOVE_TIME_C_CMD = 0x85

    DRIVE_REQUEST = 0x01
    STOP_REQUEST = 0x02
    DRIVE_TIME_REQUEST = 0x04

    ENCODER_LEN = 16

    def __init__(self) -> None:
        self.logger = logging.getLogger("i2c_controller")

        try:
            self.bus = SMBus(1)

        except OSError as e:
            self.logger.error(f"Failed to initialize I2C device! {e}")
            raise

        self.i2c_lock = threading.Lock()

        # self.logger.info("I2C Controller creating GPIO devices...")

        self.claw_tof_en = OutputDevice(20, active_high=True, initial_value=False)
        self.side_tof_en = OutputDevice(19, active_high=True, initial_value=False)
        self.front_tof_en = OutputDevice(7, active_high=True, initial_value=False)

        self.claw_tof_enabled = False
        self.side_tof_enabled = False
        self.front_tof_enabled = False

        self.init_tof()

    def init_tof(self) -> None:
        """
        Attempt to initialize all TOF sensors.

        For each TOF sensor, sets the enable pin high and attempts to
        connect via I2C. If it cannot be connected, it is marked as
        disabled. If it is connected, its I2C address is changed to a
        previously defined value.
        """
        self.adafruit_i2c = busio.I2C(board.SCL, board.SDA)

        for i in range(10):
            if not self.claw_tof_enabled:
                try:
                    self.claw_tof_en.on()
                    self.claw_tof = adafruit_vl53l1x.VL53L1X(self.adafruit_i2c)
                    self.claw_tof.set_address(self.CLAW_TOF_ADDR)
                    self.claw_tof.start_ranging()
                    self.claw_tof_enabled = True
                except Exception as e:
                    self.claw_tof_en.off()
                    self.claw_tof_enabled = False
                    self.logger.warning(f"Claw TOF init failed! {e}")

            if not self.side_tof_enabled:
                try:
                    self.side_tof_en.on()
                    self.side_tof = adafruit_vl53l1x.VL53L1X(self.adafruit_i2c)
                    self.side_tof.set_address(self.SIDE_TOF_ADDR)
                    self.side_tof.start_ranging()
                    self.side_tof_enabled = True
                except Exception as e:
                    self.side_tof_en.off()
                    self.side_tof_enabled = False
                    self.logger.warning(f"Side TOF init failed! {e}")

            if not self.front_tof_enabled:
                try:
                    self.front_tof_en.on()
                    self.front_tof = adafruit_vl53l1x.VL53L1X(self.adafruit_i2c)
                    self.front_tof.set_address(self.FRONT_TOF_ADDR)
                    self.front_tof.start_ranging()
                    self.front_tof_enabled = True
                except Exception as e:
                    self.front_tof_en.off()
                    self.front_tof_enabled = False
                    self.logger.warning(f"Front TOF init failed! {e}")

            if self.claw_tof_enabled and self.side_tof_enabled and self.front_tof_enabled:
                break

    def handle_read(self, device_address, register_address, length):
        """
        Attempt to read data over I2C.

        Connects to the device given by addr, and reads length bytes from
        location cmd. Returns either the received data, or an empty data
        array with success set to False.
        """
        addr = device_address
        cmd = register_address
        data_len = length

        response = {
            "success": False,
            "message": "",
            "data": [],
        }

        with self.i2c_lock:
            try:
                data = self.bus.read_i2c_block_data(addr, cmd, data_len)

                response["success"] = True
                response["data"] = data

            except IOError as e:
                response.message = str(e)

            return response

    def handle_write(self, device_address, register_address, data=None):
        """
        Attempt to write data over I2C.

        Connects to the device given by addr, and writes data to location
        cmd. Returns either success True or success False with an error
        message.
        """

        addr = device_address
        cmd = register_address
        if data is not None:
            write_data = data

        response = {
            "success": False,
            "message": "",
        }

        with self.i2c_lock:
            try:
                if data is not None:
                    self.bus.write_i2c_block_data(addr, cmd, write_data)
                else:
                    self.bus.write_byte(addr, cmd)

                response["success"] = True

            except IOError as e:
                response["message"] = str(e)

            return response

    def read_ultrasonic(self) -> int | None:
        """Attempt to read ultrasonic sensor data from STM32."""
        read_msg = self.handle_read(self.STM_ADDR, self.ULTRASONIC_CMD, 4)

        if not read_msg["success"]:
            self.logger.warning(f"I2C ultrasonic read failed: {read_msg['message']}")
        else:
            dist: int = int.from_bytes(read_msg["data"])

            return dist

    def read_temp(self) -> float | None:
        """Attempt to read temperature data from STM32."""
        read_msg = self.handle_read(self.STM_ADDR, self.TEMP_CMD, 4)

        if not read_msg["success"]:
            self.logger.warning(f"I2C temp read failed: {read_msg['message']}")
        else:
            temp: float = int.from_bytes(read_msg["data"]) / 100

            return temp

    def read_state(self) -> int | None:
        """Attempt to read robot state from STM32."""

        read_msg = self.handle_read(self.STM_ADDR, self.STATE_CMD, 1)

        if not read_msg["success"]:
            self.logger.warning(f"I2C state read failed: {read_msg['message']}")
        else:
            state: int = int.from_bytes(read_msg["data"])

            return state

    def read_move_time_count(self) -> int | None:
        """Attempt to read number of move time commands from STM32."""
        read_msg = self.handle_read(self.STM_ADDR, self.MOVE_TIME_C_CMD, 4)

        if not read_msg["success"]:
            self.logger.warning(f"I2C move time read failed: {read_msg['message']}")
        else:
            count: int = int.from_bytes(read_msg["data"])

            return count

    def read_tof(self, tof_name):
        """
        Attempt to read and publish data from all TOF sensors.

        For each sensor, checks if it is enabled. Then, attempts to read the
        distance data over I2C. If data is received, converts to millimetres.
        """
        with self.i2c_lock:
            if tof_name == "claw" and self.claw_tof_enabled:
                try:
                    claw_dist: float | None = self.claw_tof.distance
                except OSError:
                    claw_dist = None

                if claw_dist is None:
                    data = -1
                else:
                    data = int(claw_dist * 10)

                return data
            if tof_name == "front" and self.front_tof_enabled:
                try:
                    front_dist: float | None = self.front_tof.distance
                except OSError:
                    front_dist = None

                if front_dist is None:
                    data = -1
                else:
                    data = int(front_dist * 10)

                return data
            if tof_name == "side" and self.side_tof_enabled:
                try:
                    side_dist: float | None = self.side_tof.distance
                except OSError:
                    side_dist = None

                if side_dist is None:
                    data = -1
                else:
                    data = int(side_dist * 10)

                return data
