###########################################################################################################
# This Python script contains utility functions for the data driven forecasting workflow #
# Note: The functions are ordered alphabetically and separated by ###                                     #
###########################################################################################################

# Import required modules

# logging_config.py
import logging
from pathlib import Path

def setup_logging(log_file = 'data_driven_forecasting.log'):

    # Get the directory path of this file
    dir_path = Path(__file__).resolve().parent

    # Construct the log directory path (one level up and then into 'logs')
    log_dir = dir_path.parent / 'logs'
    
    # Create 'logs' directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # The full path to the log file
    log_filename = Path(log_dir, log_file)

    # Get the root logger
    logger = logging.getLogger()

    # Check if the logger already has handlers. If not, add a handler.
    if not logger.hasHandlers():
        # Set logging level
        logger.setLevel(logging.DEBUG)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.DEBUG)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    logging.info('Logging setup complete. Log file: {}'.format(log_filename))


