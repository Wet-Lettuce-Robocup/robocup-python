from collections import deque
import logging
from pathlib import Path

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import ImageFont


class OLEDLogHandler(logging.Handler):
    """Logging handler that sends log messages to the OLED."""

    def __init__(self, oled_controller):
        super().__init__()
        self.oled_controller = oled_controller

    def emit(self, record):
        """Send a logging record to the OLED."""

        try:
            message = self.format(record)
            self.oled_controller.add_log(
                message,
                record.levelno,
            )

        except Exception:
            self.handleError(record)


class OLEDController:
    def __init__(self):
        self.logger = logging.getLogger("oled_controller")

        self.device: ssd1306 | None = None
        self.serial: i2c | None = None

        self.status_value = "-"
        self.pi_temp_c: float | None = None
        self.stm_temp_c: float | None = None

        self.stm_temp_count = 0

        self.font_small = self._load_font(12)
        self.font_smallest = self._load_font(8)

        self.max_console_lines = 4
        self.console_lines: deque[str] = deque(maxlen=self.max_console_lines)

        self.pi_temp_path = Path("/sys/class/thermal/thermal_zone0/temp")

        try:
            self.serial = i2c(port=1, address=0x3C)
            self.device = ssd1306(self.serial, width=128, height=64, rotate=2)
            self.logger.info("OLED display initialized!")

        except OSError as e:
            self.logger.warning(f"OLED display not initialized! {e}")

        self.update_display()

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def update_display(self):
        if self.device is None:
            return
        with canvas(self.device) as draw:
            draw.text(
                (2, 0), f"Status: {self.status_value[:1]}", font=self.font_small, fill="white"
            )
            draw.text(
                (2, 12),
                self._format_temp("Pi", self.pi_temp_c),
                font=self.font_small,
                fill="white",
            )
            draw.text(
                (66, 12),
                self._format_temp("STM", self.stm_temp_c),
                font=self.font_small,
                fill="white",
            )

            # Four most recent log lines.
            padded_lines = [""] * (self.max_console_lines - len(self.console_lines)) + list(
                self.console_lines
            )
            y_positions = [26, 35, 44, 53]
            for line, y in zip(padded_lines, y_positions):
                draw.text(
                    (0, y),
                    line,
                    font=self.font_smallest,
                    fill="white",
                )

    def _truncate_line(self, text: str, max_len: int = 28) -> str:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    def add_logs(self, message: str, level: int = logging.INFO) -> None:
        """
        Add a log message to the OLED.

        :param message: Message to display.
        :param level: Python logging level.
        """
        if self.current_page == 1:
            self.update_display()

        if level >= logging.CRITICAL:
            level_letter = "F"
        elif level >= logging.ERROR:
            level_letter = "E"
        elif level >= logging.WARNING:
            level_letter = "W"
        elif level >= logging.INFO:
            level_letter = "I"
        else:
            level_letter = "D"

        line = f"{level_letter} {message}"
        self.console_lines.append(self._truncate_line(line))

        self.update_display()

    def create_log_handler(self) -> logging.Handler:
        """Create a logging handler that sends logs to the OLED. Attach the returned handler to the robot logger."""
        handler = OLEDLogHandler(self)

        # Only show the actual log message on the OLED.
        handler.setFormatter(logging.Formatter("%(message)s"))

        return handler

    def get_pi_temp(self):
        self.pi_temp_c = self._read_pi_temp_c()

    def _read_pi_temp_c(self) -> float | None:
        try:
            raw = self.pi_temp_path.read_text(encoding="utf-8").strip()
            return float(raw) / 1000.0
        except (OSError, ValueError):
            return None

    def _format_temp(self, label: str, value: float | None) -> str:
        if value is None:
            return f"{label}: --.-C"
        return f"{label}: {value:.1f}C"

    def set_status(self, msg: str):
        self.status_value = msg.strip() or "-"
        self.update_display()

    def get_stm_temp(self, msg: float):
        self.stm_temp_count += 1
        if self.stm_temp_count == 5:
            self.stm_temp_count = 0
            self.stm_temp_c = msg

            self.update_display()

    def cleanup(self) -> None:
        """Clear the display."""
        if self.device is not None:
            self.device.clear()
