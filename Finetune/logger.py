import logging
import datetime


def setup_logger():

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")

    file_handler = logging.FileHandler(f'logFile_{current_datetime}.log')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()