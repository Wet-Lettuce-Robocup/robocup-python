import math

from gpiozero import OutputDevice


class ServoController:
    """
    Handles controlling a servo through sending commands to STM32 over I2C, and enabling
    servos through a GPIO pin connected to a mosfet in series with the servo power.


        All angles are in radians.

    """

    def __init__(self, servo_id=0, stm_address=0x67, servo_cmd=0x10, gpio_pin=1) -> None:
        self.servo_id: int = servo_id
        self.stm_address: int = stm_address
        self.servo_cmd: int = servo_cmd
        self.gpio_pin: int = gpio_pin

        self.gpio_device = OutputDevice(self.gpio_pin, active_high=True, initial_value=False)

    @staticmethod
    def rads_to_degrees(angle: float) -> float:
        """
        Convert radians to degrees.

        :param angle: Angle in radians.
        :type angle: float

        :returns: Angle in degrees.
        :rtype: float
        """
        return angle * 180 / math.pi

    def servo_callback(self, request):
        """
        Set servo position.

        Receives servo position in radians, and converts it to degrees
        to send to the STM32 over I2C.

        :param msg: Angle to set servo to in radians.
        :type msg: Float32
        """
        degrees = int(self.rads_to_degrees(request.angle))
        degrees = min(max(degrees, 0), 180) & 0xFF

        i2c_request = I2CWrite.Request()

        i2c_request.device_address = self.i2c_address
        i2c_request.register_address = self.servo_cmd
        i2c_request.data = [self.servo_id, degrees]

        # self.get_logger().info(f'Servo {self.service_name} called with angle {degrees}')

        try:
            self.gpio_device.on()
            future = self.cli.call_async(i2c_request)
            i2c_response = await future

            if i2c_response is None:
                response.success = False
                response.message = "No response from I2C service"
                return response

            if not i2c_response.success:
                response.success = False
                response.message = i2c_response.message
                return response

            response.success = True
            response.message = ""

        except Exception as e:
            response.success = False
            response.message = str(e)

            self.get_logger().error(f"Servo command failed: {e}")

        return response

    def cleanup(self) -> None:
        """Turn off servo on node exit."""
        self.gpio_device.off()
