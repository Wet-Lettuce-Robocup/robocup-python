import os


class PWMController:
    def __init__(self, pwm_chip=0, pwm_channel=0) -> None:
        self.pwm_chip = pwm_chip
        self.pwm_channel = pwm_channel

        self.chip_path = os.path.join("/sys/class/pwm", f"pwmchip{self.pwm_chip}")
        self.channel_path = os.path.join(self.chip_path, f"pwm{self.pwm_channel}")

        if os.path.isdir(self.channel_path):
            with open(os.path.join(self.channel_path, "enable"), "w") as f:
                f.write("0")

        else:
            with open(os.path.join(self.chip_path, "export"), "w") as f:
                f.write(str(self.pwm_channel))

        self.enable_on_set_period: bool = False

        self.period = 0

    def set_enable(self, msg: bool) -> None:
        """
        Enable or disable the PWM channel.

        First checks if the period has been set. If not, does not enable to
        prevent errors (writing 1 to enable throws an error if period is 0),
        but sets enable_on_set_period to enable the channel automatically
        once the period has been set. If the period has been set, writes data
        to enable.

        :param msg: Data to write to enable.
        :type msg: Bool
        """

        if self.period == 0:
            self.enable_on_set_period = True
            return

        with open(os.path.join(self.channel_path, "enable"), "w") as f:
            f.write(str(int(msg)))

    def set_period(self, msg: int) -> None:
        """
        Set the desired period of the PWM channel.

        First sets the period as specified. Then checks if the
        enable_on_set_period variable has been set to True, in which cas it
        will also enable the channel.
        """
        with open(os.path.join(self.channel_path, "period"), "w") as f:
            f.write(str(msg))
            self.period = msg

        if not self.enable_on_set_period:
            return

        with open(os.path.join(self.channel_path, "enable"), "w") as f:
            f.write("1")
            self.enable_on_set_period = False

    def set_duty_cycle(self, msg: int) -> None:
        """Set the desired duty cycle of the PWM channel."""
        duty_cycle = msg
        duty_cycle = min(duty_cycle, self.period)

        with open(os.path.join(self.channel_path, "duty_cycle"), "w") as f:
            f.write(str(duty_cycle))
