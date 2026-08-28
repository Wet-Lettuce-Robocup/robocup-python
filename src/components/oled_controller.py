from collections import deque
from pathlib import Path

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import ImageFont


class OLEDController:
    def __init__(self):
        self.device: ssd1306 | None = None
        self.serial: i2c | None = None

        self.status_value = "-"
        self.pi_temp_c: float | None = None
        self.stm_temp_c: float | None = None

        self.stm_temp_count = 0

        self.font_large = self._load_font(36)
        self.font_small = self._load_font(12)
        self.font_smallest = self._load_font(8)

        self.max_console_lines = 6
        self.console_lines: deque[str] = deque(maxlen=self.max_console_lines)
        self.pi_temp_path = Path("/sys/class/thermal/thermal_zone0/temp")

        try:
            self.serial = i2c(port=1, address=0x3C)
            self.device = ssd1306(self.serial, width=128, height=64, rotate=2)
            print("OLED display initialized!")

        except OSError as e:
            print(f"OLED display not initialized! {e}")

        if self.device is None:
            return

        self.update_display()

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def update_display(self):
        with canvas(self.device) as draw:
            draw.text((2, 6), self.status_value[:1], font=self.font_large, fill="white")
            draw.text(
                (2, 40),
                self._format_temp("Pi", self.pi_temp_c),
                font=self.font_small,
                fill="white",
            )
            draw.text(
                (2, 52),
                self._format_temp("STM", self.stm_temp_c),
                font=self.font_small,
                fill="white",
            )

            lines = self.console_lines
            padded = deque([""]) * max(0, 6 - len(lines)) + lines
            y_positions = [0, 10, 20, 30, 40, 50]
            for line, y in zip(padded, y_positions):
                draw.text((0, y), line, font=self.font_smallest, fill="white")

    def _truncate_line(self, text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    def get_pi_temp(self):
        self.pi_temp_c = self._read_pi_temp_c()

    def get_logs(self, msg):
        """Idk howi to do this"""
        level = self._level_to_letter(msg.level)
        line = f"{level} {msg.msg}"
        self.console_lines.append(self._truncate_line(line, 45))
        if self.current_page == 1:
            self.update_display()

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

    def status_callback(self, msg: String):
        self.status_value = msg.data.strip() or "-"
        if self.current_page == 0:
            self.update_display()

    def error_callback(self, msg: Float64):
        self.error_value = int(msg.data)
        if self.current_page == 0:
            self.update_display()

    def silver_callback(self, msg: Int32):
        self.silver_value = msg.data
        if self.current_page == 0:
            self.update_display()

    def black_callback(self, msg: Int32):
        self.black_value = msg.data
        if self.current_page == 0:
            self.update_display()

    def stm_temp_callback(self, msg: Float32):
        self.stm_temp_count += 1
        if self.stm_temp_count == 5:
            self.stm_temp_count = 0
            self.stm_temp_c = msg.data
            if self.current_page == 0:
                self.update_display()
