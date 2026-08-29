import logging
from pathlib import Path


def setup_logging(
    log_file: str = "logs/latest.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure logging for the robot.

    ```
    Outputs logs to both the terminal and a file.

    :param log_file: Path to the log file.
    :param level: Minimum logging level.
    :return: Root robot logger.
    """

    logger = logging.getLogger("robot")
    logger.setLevel(level)

    # Prevent duplicate messages if setup_logging() is called again.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Terminal output.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File output.
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
