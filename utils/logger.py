import logging
import os

def get_logger(config):
    log_file = config["file"]
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("CryptoSMCAlgo")
    logger.setLevel(getattr(logging, config["level"]))
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger