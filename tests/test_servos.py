from time import sleep

from components.i2c_controller import I2CBusController
from components.servo_controller import ServoController


class TestServos:
    def __init__(self):
        self.i2c_controller = I2CBusController()
        self.servo_grab = ServoController(self.i2c_controller, servo_id=0, gpio_pin=4)
        self.servo_lift = ServoController(self.i2c_controller, servo_id=1, gpio_pin=0)
        self.servo_tray_release = ServoController(self.i2c_controller, servo_id=2, gpio_pin=1)

    def disable_servos(self) -> None:
        """Disable power to the servos."""
        write_response = self.i2c_controller.handle_write(0x67, 0x14)

        if not write_response["success"]:
            self.logger.error(f"Servo stop command failed: {write_response['message']}")
            return False

        return True

    def test(self):
        for i in range(3, 10):
            angle = 10 * i
            self.servo_grab.set_angle(angle)
            print(angle)
            sleep(1)


run = TestServos()
try:
    run.test()
except KeyboardInterrupt:
    pass
finally:
    run.disable_servos()
