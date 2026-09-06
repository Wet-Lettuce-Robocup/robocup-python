import logging
import threading
from enum import Enum

from gpiozero import Button

from src.components.i2c_controller import I2CBusController
from src.components.oled_controller import OLEDController
from src.components.robot_logging import setup_logging
from src.line_follow import Follow
from src.rescue import Rescue
from src.robot import Robot


class Task(Enum):
    INIT = 1
    IDLE = 2
    FOLLOW = 3
    RESCUE = 4


class Main:
    def __init__(self):

        self.logger = setup_logging()

        self.oled = OLEDController()
        self.oled_handler = self.oled.create_log_handler()
        self.oled_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.oled_handler)

        self.i2c_controller = I2CBusController()

        self.robot = Robot(self.i2c_controller)

        self.rescue = Rescue(self.i2c_controller, self.robot)
        self.follow = Follow(self.i2c_controller, self.robot)

        self.button = Button(6, pull_up=True)
        self.button.when_pressed = self._on_pressed

        self.stop_event = threading.Event()

        self.current_task = Task.INIT
        self.task_started = False

    def _transition_to(self, task):
        """Change to a new rescue task."""
        self.logger.info(f"Task: {self.current_task.name} -> {task.name}")
        self.current_task = task
        self.task_started = False

    def _on_pressed(self):
        self.stop_event.set()
        self.stop()

    def reset_stop(self):
        self.stop_event.clear()

    def main(self):

        self.logger.info("Robot started")

        self.logger.warning("test warning")
        self.logger.error("test error")

        while True:
            if self.current_task == Task.INIT:
                if self.task_started:
                    return
                self.task_started = True
                self._transition_to(Task.IDLE)
            elif self.current_task == Task.IDLE:
                self.robot.reset_stop()
                rescue_thread = threading.Thread(target=self.rescue, daemon=True)

                rescue_thread.start()

    def rescue(self):
        while not self.robot.stop_event.is_set():
            pass

    def follow(self):
        while not self.robot.stop_event.is_set():
            pass
