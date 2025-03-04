"""
Logging utility module.
"""
import logging
import os
import datetime

def setup_logger(log_file=None, level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Args:
        log_file (str, optional): Path to the log file
        level (int, optional): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_timestamped_log_file(base_dir='logs', prefix='workflow'):
    """
    Generate a timestamped log file path.
    
    Args:
        base_dir (str, optional): Base directory for logs
        prefix (str, optional): Prefix for the log file name
        
    Returns:
        str: Path to the log file
    """
    # Create directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create log file path
    log_file = os.path.join(base_dir, f'{prefix}_{timestamp}.log')
    
    return log_file 