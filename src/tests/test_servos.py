from time import monotonic as sleep

from components.i2c_controller import I2CBusController
from components.servo_controller import ServoController


class TestServos:
    def __init__(self):
        self.i2c_controller = I2CBusController()
        self.servo_grab = ServoController(self.i2c_controller, servo_id=0, gpio_pin=4)
        self.servo_lift = ServoController(self.i2c_controller, servo_id=1, gpio_pin=0)
        self.servo_tray_release = ServoController(self.i2c_controller, servo_id=2, gpio_pin=1)

    def relax_servos(self):
        self.servo_grab.disable()
        self.servo_lift.disable()
        self.servo_tray_release.disable()

    def cleanup_servos(self):
        self.servo_grab.cleanup()
        self.servo_lift.cleanup()
        self.servo_tray_release.cleanup()

    def test(self):
        self.servo_grab.set_angle(54)
        sleep(1)
        self.relax_servos()


run = TestServos()
try:
    run.test()
except Exception as e:
    print(str(e))
finally:
    run.cleanup_servos()
