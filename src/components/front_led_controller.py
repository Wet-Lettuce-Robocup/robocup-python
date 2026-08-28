from src.components.pwm_controller import PWMController


class LEDController:
    PWM_CHANNEL = 0
    PERIOD = 100000

    def __init__(self) -> None:
        self.pwm_controller = PWMController(pwm_channel=self.PWM_CHANNEL)
        self.target_brightness = 0

        self.set_brightness(0)

    def set_brightness(self, target_brightness: int) -> None:
        print(f"Setting LED brightness to {target_brightness}%")
        if target_brightness < 0 or target_brightness > 100:
            print("Target brightness must be between 0 and 100!")
            return

        self.target_brightness = target_brightness

        if target_brightness == 0:
            self.pwm_controller.set_enable(False)
            return

        self.pwm_controller.set_period(self.PERIOD)

        # convert percentage to duty cycle (0-100000) for 100kHz period
        self.pwm_controller.set_duty_cycle(int((target_brightness / 100) * self.PERIOD))

        self.pwm_controller.set_enable(True)

        return
