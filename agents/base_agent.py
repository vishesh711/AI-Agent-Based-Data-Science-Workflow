"""
Base Agent module.
"""
import logging
import time
import pandas as pd

class BaseAgent:
    """
    Base class for all agents in the workflow.
    
    Provides common functionality for logging, timing, and data validation.
    """
    
    def __init__(self, name):
        """
        Initialize the Base Agent.
        
        Args:
            name (str): Name of the agent
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.start_time = None
    
    def log_start(self):
        """Log the start of processing."""
        self.start_time = time.time()
        self.logger.info(f"Starting {self.name} processing")
    
    def log_end(self):
        """Log the end of processing with elapsed time."""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            self.logger.info(f"Completed {self.name} processing in {elapsed_time:.2f} seconds")
        else:
            self.logger.info(f"Completed {self.name} processing")
    
    def validate_input(self, data):
        """
        Validate the input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data is None:
            self.logger.error("Input data is None")
            return False
        
        if not isinstance(data, pd.DataFrame):
            self.logger.error(f"Input data is not a pandas DataFrame, got {type(data)}")
            return False
        
        if data.empty:
            self.logger.error("Input data is empty")
            return False
        
        return True
    
    def process(self, data, **kwargs):
        """
        Process the data.
        
        This method should be overridden by subclasses.
        
        Args:
            data: Input data to process
            **kwargs: Additional parameters for processing
            
        Returns:
            tuple: (processed_data, report)
        """
        raise NotImplementedError("Subclasses must implement process method") 