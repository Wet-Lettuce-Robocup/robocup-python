import logging

from src.components.robot_logging import setup_logging
from src.components.oled_controller import OLEDController

logger = setup_logging()


oled = OLEDController()

oled_handler = oled.create_log_handler()
oled_handler.setLevel(logging.INFO)

logger.addHandler(oled_handler)

logger.info("Robot started")

logger.info("test info")
logger.warning("test warning")
logger.error("test error")
