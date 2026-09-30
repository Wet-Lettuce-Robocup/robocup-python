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
        self.servo_grab.set_angle(30)
        sleep(1)
        self.servo_grab.set_angle(40)
        sleep(1)
        self.servo_grab.set_angle(50)
        sleep(1)
        self.servo_grab.set_angle(60)
        sleep(1)
        self.servo_grab.set_angle(70)
        sleep(1)
        self.servo_grab.set_angle(80)
        sleep(1)
        self.servo_grab.set_angle(90)
        sleep(1)
        self.servo_grab.set_angle(100)
        sleep(1)


run = TestServos()
try:
    run.test()
except Exception as e:
    print(str(e))
finally:
    run.disable_servos()
