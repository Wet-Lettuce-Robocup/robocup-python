import logging
import threading
import time
from enum import Enum

import cv2
from gpiozero import Button

from components.fan_controller import FanController
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

    BUTTON_DEBOUNCE_TIME = 0.10

    VNC = False
    DEBUG = True
    VIDEO = False

    def __init__(self):

        self.logger = setup_logging()

        self.oled = OLEDController()
        self.oled_handler = self.oled.create_log_handler()
        self.oled_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.oled_handler)

        self.i2c_controller = I2CBusController()

        self.robot = Robot(self.i2c_controller)
        self.fan = FanController()

        self.rescue = Rescue(self.i2c_controller, self.robot)
        self.follow = Follow(self.i2c_controller, self.robot)

        self.button = Button(6, pull_up=True, bounce_time=self.BUTTON_DEBOUNCE_TIME)

        self.button.when_released = self._on_pressed

        self.button_event = threading.Event()

        self.last_button_time = 0.0

        self.stop_event = threading.Event()

        self.current_task = Task.INIT
        self.task_started = False

        self.rescue_thread = None
        self.follow_thread = None

        self.target_rescue_loop_time = None
        self.target_line_follow_loop_time = None

        if self.VIDEO and self.DEBUG:
            self.followFile = cv2.VideoWriter(
                "followOutput.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 100)
            )
            self.rescueFile = cv2.VideoWriter(
                "rescueOutput.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, (1536, 864)
            )
        else:
            self.followFile = None
            self.rescueFile = None

    def _transition_to(self, task):
        if self.current_task == task:
            return

        self.logger.info(f"Main task: {self.current_task.name} -> {task.name}")

        self.current_task = task
        self.task_started = False

    def _on_pressed(self):

        now = time.monotonic()

        if now - self.last_button_time < self.BUTTON_DEBOUNCE_TIME:
            return

        self.last_button_time = now

        self.button_event.set()

    def _handle_button(self):
        """
        Process a button press from the main application thread.

        All task state changes happen here, rather than inside the GPIO callback.
        """

        if not self.button_event.is_set():
            return

        self.button_event.clear()

        self.logger.info("Button pressed")

        if self.current_task == Task.IDLE:
            self.logger.info("Starting line follow")

            self.stop_event.clear()

            self._transition_to(Task.FOLLOW)

        elif self.current_task == Task.FOLLOW:
            self.logger.info("Stopping line follow")

            self.stop_event.set()

            self.robot.stop_moving()

        elif self.current_task == Task.RESCUE:
            self.logger.info("Stopping rescue")

            self.stop_event.set()

            self.robot.stop_moving()

    def reset_stop(self):
        self.stop_event.clear()

    def main(self):

        self.logger.info("Robot started")

        self.fan.manual_fan_speed(60)

        while True:
            try:
                # Handle button presses from the main thread.
                self._handle_button()

                if self.current_task == Task.INIT:
                    if self.task_started:
                        pass

                    self.task_started = True

                    self.robot.claw("grab")
                    self.robot.lift("up")
                    self.robot.tray("reset")

                    self._transition_to(Task.IDLE)

                elif self.current_task == Task.IDLE:
                    pass

                elif self.current_task == Task.RESCUE:
                    if not self.task_started:
                        self.task_started = True
                        self.reset_stop()

                        self.rescue.reset()

                        self.rescue_thread = threading.Thread(
                            target=self.rescue_loop,
                            daemon=True,
                            name="RescueThread",
                        )

                        self.rescue_thread.start()

                    # Debug display
                    if self.DEBUG:
                        debug_frame = self.rescue.get_debug_frame()

                        if debug_frame is not None:
                            if self.VNC:
                                cv2.imshow("Debug", debug_frame)
                                cv2.waitKey(1)
                            elif self.VIDEO:
                                self.rescueFile.write(debug_frame)

                    # Wait until the worker has stopped
                    if not self.rescue_thread.is_alive():
                        cv2.destroyAllWindows()
                        if self.rescue.is_finished():
                            self.logger.info("Rescue finished")
                            self._transition_to(Task.FOLLOW)
                        else:
                            self._transition_to(Task.IDLE)
                        self.rescue.reset()

                    time.sleep(0.05)

                elif self.current_task == Task.FOLLOW:
                    if not self.task_started:
                        self.task_started = True
                        self.reset_stop()

                        self.follow.reset()

                        self.follow_thread = threading.Thread(
                            target=self.follow_loop,
                            daemon=True,
                            name="FollowThread",
                        )

                        self.follow_thread.start()

                    # Debug display
                    if self.DEBUG:
                        debug_frame = self.follow.get_debug_frame()

                        if debug_frame is not None:
                            if self.VNC:
                                cv2.imshow("Debug", debug_frame)
                                cv2.waitKey(1)
                            elif self.VIDEO:
                                self.followFile.write(debug_frame)

                    if not self.follow_thread.is_alive():
                        cv2.destroyAllWindows()
                        if self.follow.is_finished():
                            self.logger.info("Line follow finished")
                            self._transition_to(Task.RESCUE)
                        else:
                            self._transition_to(Task.IDLE)
                        self.follow.reset()

                    time.sleep(0.05)

            except Exception as e:
                self.logger.error(f"Caught an error in main loop! {e}")

    def rescue_loop(self):
        try:
            while not self.stop_event.is_set():
                self.target_rescue_loop_time = time.monotonic() + (
                    1 / self.RESCUE_LOOPS_PER_SECOND
                )

                self.rescue.tick_rescue()

                if self.rescue.is_finished():
                    break

                now = time.monotonic()

                if now < self.target_rescue_loop_time:
                    time.sleep(self.target_rescue_loop_time - now)

            self.robot.stop_moving()

            self.ball_tray_memory = self.rescue.exit()

            self.logger.info("Rescue stopped")
        except Exception:
            self.logger.exception("Exception in rescue")
            self.robot.stop_moving()

    def follow_loop(self):
        try:
            while not self.stop_event.is_set():
                self.target_line_follow_loop_time = time.monotonic() + (
                    1 / self.FOLLOW_LOOPS_PER_SECOND
                )

                self.follow.main()

                if self.follow.is_finished():
                    break

                now = time.monotonic()

                if now < self.target_line_follow_loop_time:
                    time.sleep(self.target_line_follow_loop_time - now)

            self.robot.stop_moving()

            self.logger.info("Line follow stopped")
        except Exception:
            self.logger.exception("Exception in line follow")
            self.robot.stop_moving()

    def cleanup(self):
        self.robot.stop_moving()
        self.button.close()
        self.fan.manual_fan_speed(0)
        self.i2c_controller.front_tof_en.close()
        self.i2c_controller.side_tof_en.close()
        self.i2c_controller.claw_tof_en.close()

        cv2.destroyAllWindows()


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
