from time import sleep

from components.i2c_controller import I2CBusController
from robot import Robot


class TestServos:
    def __init__(self):
        self.i2c_controller = I2CBusController()

        self.robot = Robot(self.i2c_controller)

    def disable_servos(self) -> None:
        """Disable power to the servos."""
        write_response = self.i2c_controller.handle_write(0x67, 0x14)

        if not write_response["success"]:
            self.logger.error(f"Servo stop command failed: {write_response['message']}")
            return False

        return True

    def test(self):
        self.robot.tray("reset")
        sleep(0.5)
        self.robot.claw("grab")
        sleep(1)
        self.robot.lift("up")
        sleep(1)

        self.robot.claw("release")
        sleep(1)
        self.robot.claw("grab")
        sleep(1)

        self.robot.lift("down")
        sleep(1)

        self.robot.tray("release")
        sleep(1)
        self.robot.tray("reset")


run = TestServos()
try:
    run.test()
except KeyboardInterrupt:
    pass
finally:
    run.disable_servos()
