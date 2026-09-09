import logging
from datetime import datetime
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
        datefmt="%H:%M:%S",
    )

    # Terminal output.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File output.
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8") as file:
                first_line = file.readline().strip()

            if first_line.startswith("Created: "):
                timestamp_string = first_line.removeprefix("Created: ")
                created = datetime.strptime(
                    timestamp_string,
                    "%Y-%m-%d %H:%M:%S",
                )

                archive_name = (
                    f"{log_path.stem}_{created.strftime('%Y%m%d_%H%M%S')}{log_path.suffix}"
                )
            else:
                # Fallback if the timestamp is invalid
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"{log_path.stem}_{timestamp}{log_path.suffix}"

            archive_path = log_path.with_name(archive_name)

            # Avoid overwriting existing logs
            counter = 1
            while archive_path.exists():
                archive_path = log_path.with_name(
                    f"{log_path.stem}_{created.strftime('%Y%m%d_%H%M%S')}"
                    f"_{counter}{log_path.suffix}"
                )
                counter += 1

            log_path.rename(archive_path)

        except (OSError, ValueError):
            # If the old log cannot be read/renamed, continue and let FileHandler add onto the latest log
            pass

    # Create the new latest.log and add its timestamp
    created = datetime.now()

    with log_path.open("w", encoding="utf-8") as file:
        file.write(f"Created: {created.strftime('%Y-%m-%d %H:%M:%S')}\n")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
