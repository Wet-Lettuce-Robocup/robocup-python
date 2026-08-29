import logging
import math

from gpiozero import OutputDevice


class ServoController:
    """
    Controls a servo through the STM32 over I2C.

    Angles supplied to set_angle() are in radians.
    """

    def __init__(
        self,
        i2c_controller,
        servo_id: int,
        i2c_address: int = 0x67,
        servo_cmd: int = 0x10,
        gpio_pin: int = 1,
    ) -> None:

        self.logger = logging.getLogger("servo_controller")

        self.i2c_controller = i2c_controller

        self.servo_id = servo_id & 0xFF
        self.i2c_address = i2c_address & 0x7F
        self.servo_cmd = servo_cmd & 0xFF

        self.gpio_device = OutputDevice(
            gpio_pin,
            active_high=True,
            initial_value=False,
        )

    @staticmethod
    def rads_to_degrees(angle: float) -> float:
        """Convert radians to degrees."""
        return angle * 180.0 / math.pi

    def set_angle(self, angle: float) -> bool:
        """
        Set the servo position.

        :param angle: Servo angle in radians.
        :return: True if the command was sent successfully.
        """

        degrees = int(self.rads_to_degrees(angle))
        degrees = max(0, min(degrees, 180))

        # Enable servo power
        self.gpio_device.on()

        write_response = self.i2c_controller.handle_write(
            self.i2c_address,
            self.servo_cmd,
            [self.servo_id, degrees],
        )

        if not write_response["success"]:
            self.logger.error(f"Servo {self.servo_id} command failed: {write_response['message']}")
            return False

        return True

    def disable(self) -> None:
        """Disable power to the servo."""
        self.gpio_device.off()

    def cleanup(self) -> None:
        """Clean up the GPIO device."""
        self.gpio_device.off()
        self.gpio_device.close()
