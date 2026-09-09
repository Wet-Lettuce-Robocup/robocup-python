import logging
import threading
import time
from enum import Enum

from components import i2c_controller
from gpiozero import Button

from components.i2c_controller import I2CBusController
from components.oled_controller import OLEDController
from components.robot_logging import setup_logging
from line_follow import Follow
from rescue import Rescue
from robot import Robot


class Task(Enum):
    INIT = 1
    IDLE = 2
    FOLLOW = 3
    RESCUE = 4


class Main:
    RESCUE_LOOPS_PER_SECOND = 10
    FOLLOW_LOOPS_PER_SECOND = 20

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
        self.button.when_released = self._on_pressed

        self.stop_event = threading.Event()

        self.current_task = Task.INIT
        self.task_started = False

        self.rescue_thread = None
        self.follow_thread = None

        self.target_rescue_loop_time = None
        self.target_line_follow_loop_time = None

    def _transition_to(self, task):
        self.logger.info(f"Task: {self.current_task.name} -> {task.name}")

        self.current_task = task
        self.task_started = False

    def _on_pressed(self):
        """Called automatically when the button is pressed."""

        self.logger.info("Button pressed")

        # If something is currently running, stop it
        if self.current_task == Task.RESCUE:
            self.logger.info("Stopping rescue")
            self.robot.stop_moving()
            self.stop_event.set()

        elif self.current_task == Task.FOLLOW:
            self.logger.info("Stopping line follow")
            self.robot.stop_moving()
            self.stop_event.set()

        # If in idle, start line following.
        elif self.current_task == Task.IDLE:
            self.logger.info("Starting line follow")
            self._transition_to(Task.FOLLOW)

    def reset_stop(self):
        self.stop_event.clear()

    def main(self):

        self.logger.info("Robot started")

        while True:
            try:
                if self.current_task == Task.INIT:
                    if self.task_started:
                        time.sleep(0.01)
                        continue

                    self.task_started = True
                    self._transition_to(Task.IDLE)

                elif self.current_task == Task.IDLE:
                    time.sleep(0.01)

                elif self.current_task == Task.RESCUE:
                    if not self.task_started:
                        self.task_started = True
                        self.reset_stop()

                        self.rescue_thread = threading.Thread(target=self.rescue_loop, daemon=True)

                        self.rescue_thread.start()

                    # Check whether rescue has finished
                    elif not self.rescue_thread.is_alive():
                        if self.rescue.is_finished():
                            self.logger.info("Rescue finished")
                            self._transition_to(Task.FOLLOW)
                        else:
                            self._transition_to(Task.IDLE)

                    time.sleep(0.01)

                elif self.current_task == Task.FOLLOW:
                    if not self.task_started:
                        self.task_started = True
                        self.reset_stop()

                        self.follow_thread = threading.Thread(target=self.follow_loop, daemon=True)

                        self.follow_thread.start()

                    if not self.follow_thread.is_alive():
                        if self.follow.is_finished():
                            self.logger.info("Line follow finished")
                            self._transition_to(Task.RESCUE)
                        else:
                            self._transition_to(Task.IDLE)

                    time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Caught an error in main loop! {e}")

    def rescue_loop(self):
        while not self.stop_event.is_set():
            self.target_rescue_loop_time = time.monotonic() + (1 / self.RESCUE_LOOPS_PER_SECOND)

            self.rescue.tick_rescue()

            now = time.monotonic()
            if now < self.target_rescue_loop_time:
                time.sleep(self.target_rescue_loop_time - now)

        self.robot.stop_moving()
        self.logger.info("Rescue stopped")

    def follow_loop(self):
        while not self.stop_event.is_set():
            self.target_follow_loop_time = time.monotonic() + (1 / self.FOLLOW_LOOPS_PER_SECOND)

            self.follow.main()

            now = time.monotonic()
            if now < self.target_follow_loop_time:
                time.sleep(self.target_follow_loop_time - now)

        self.robot.stop_moving()
        self.logger.info("Line follow stopped")

    def cleanup(self):
        self.robot.stop_moving()
        self.button.close()
        self.i2c_controller.front_tof_en.close()
        self.i2c_controller.side_tof_en.close()
        self.i2c_controller.claw_tof_en.close()


if __name__ == "__main__":
    runtime = None
    try:
        runtime = Main()
        runtime.main()
    except KeyboardInterrupt:
        pass
    finally:
        if runtime is not None:
            runtime.cleanup()
