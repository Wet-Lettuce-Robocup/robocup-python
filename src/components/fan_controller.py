import logging
from math import isclose
from pathlib import Path
import time

from gpiozero import DigitalInputDevice
from src.components.pwm_controller import PWMController


class FanController:
    """
    Node for controlling fan.

    - Subscribed to fan/target_speed (Int32, 0-100%)
    - RPM is sampled over 3 seconds and published as percentage to fan/speed (Int32, 0-100%)
    """

    PWM_CHANNEL = 2
    PERIOD = 100000
    TACH_PIN = 11
    PULSES_PER_REV = 2
    MAX_RPM = 2000  # min 450

    IDLE_SPEED = 40
    LOW_TEMP, LOW_SPEED = 40, 50
    MED_TEMP, MED_SPEED = 50, 70
    HIGH_TEMP, HIGH_SPEED = 70, 100

    def __init__(self) -> None:
        self.logger = logging.getLogger("fan_controller")

        self.last_auto_speed = 0
        self.auto_enabled = True
        self.pi_temp_path = Path("/sys/class/thermal/thermal_zone0/temp")

        # setup tachometer input
        self.tach = DigitalInputDevice(self.TACH_PIN)

        self.target_speed = 0
        self.current_speed = 0
        self.last_set = time.now()

        self.count = 0
        self.tach.when_activated = self.tach_interrupt

        self.pwm_controller = PWMController(pwm_channel=self.PWM_CHANNEL)

    def run_fan(self):
        self.pi_temp_c = self._read_pi_temp_c()

        if self.pi_temp_c > self.HIGH_TEMP:
            spd = self.HIGH_SPEED
        if self.pi_temp_c > self.MED_TEMP:
            spd = self.MED_SPEED
        if self.pi_temp_c > self.LOW_TEMP:
            spd = self.LOW_SPEED
        else:
            spd = self.IDLE_SPEED

        if spd != self.last_auto_speed and self.auto_enabled:
            self.logger.info(f"Automatically setting fan target to {spd}%")
            self.set_fan_speed(spd)

        self.last_auto_speed = spd

    def _read_pi_temp_c(self) -> float | None:
        try:
            raw = self.pi_temp_path.read_text(encoding="utf-8").strip()
            return float(raw) / 1000.0
        except (OSError, ValueError):
            return None

    def manual_fan_speed(self, msg) -> None:
        self.auto_enabled = False

        target_speed = msg
        self.logger.info(f"Manually setting fan target to {target_speed}%")
        self.set_fan_speed(target_speed)

    def set_fan_speed(self, target_speed: int) -> None:
        if target_speed < 0 or target_speed > 100:
            self.logger.warning("Target speed must be between 0 and 100!")
            return

        self.target_speed = target_speed
        self.last_set = time.now()

        if target_speed == 0:
            self.pwm_controller.set_enable(False)
            return

        # Set period pin to self.PERIOD
        self.pwm_controller.set_period(self.PERIOD)

        # convert percentage to duty cycle (0-self.PERIOD) for 100kHz period
        self.pwm_controller.set_duty_cycle(int((target_speed / 100) * self.PERIOD))

        self.pwm_controller.set_enable(True)
        return

    def tach_interrupt(self):
        self.count += 1

    def calculate_speed(self) -> None:
        freq = self.get_frequency()
        rpm = (freq * 60) / self.PULSES_PER_REV
        time_now = time.now()
        dt = (time_now - self.last_set).nanoseconds
        if dt > 5e9 and self.target_speed != 0 and rpm <= 1:
            # if fan is stopped, set target to 0 to prevent excess current draw
            self.set_fan_speed(0)

            self.logger.warning("Fan is stalling, disabling fan")
        # convert to percentage of max speed (2000 RPM)
        self.current_speed = int((rpm / self.MAX_RPM) * 100)

    def get_frequency(self) -> int:
        hz = self.count / 3.0  # 3s for Hz
        self.count = 0
        return int(hz)

    def check_working(self) -> None:
        # for debugging, check if fan is working by comparing target and current speed
        self.calculate_speed()
        if self.current_speed > 0:
            if isclose(self.current_speed, self.target_speed, abs_tol=10):
                self.logger.info("Target is close to current speed, fan is working")
            else:
                self.logger.warning("Target is not close to current speed")
        else:
            self.logger.info("Fan speed is 0")
        self.logger.info(
            f"target speed: {self.target_speed}% | current speed: {self.current_speed}%"
        )
