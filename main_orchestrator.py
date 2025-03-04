"""
Main Orchestrator module.
"""
import pandas as pd
import logging
import os
from workflow import Workflow
import json
from utils.llm_api import LLMProvider

class MainOrchestrator:
    """
    Acts as the entry point of the system.
    
    Loads the dataset and calls the workflow execution module.
    """
    
    def __init__(self, data_path=None, data=None, use_llm=False, llm_provider="openai"):
        """
        Initialize the Main Orchestrator.
        
        Args:
            data_path (str, optional): Path to the dataset file
            data (pd.DataFrame, optional): DataFrame to use directly
            use_llm (bool): Whether to use LLM for enhanced insights
            llm_provider (str): LLM provider to use ('openai' or 'groq')
        """
        # Configure logging
        self._configure_logging()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workflow = Workflow(use_llm=use_llm, llm_provider=llm_provider)
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        
        # Load data if provided
        self.data = None
        if data_path:
            self.load_data(data_path)
        elif isinstance(data, pd.DataFrame):
            self.data = data
    
    def _configure_logging(self):
        """Configure logging for the system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('data_science_workflow.log')
            ]
        )
    
    def load_data(self, data_path):
        """
        Load data from the specified path.
        
        Args:
            data_path (str): Path to the dataset file
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        self.logger.info(f"Loading data from {data_path}")
        
        try:
            # Determine file extension
            _, ext = os.path.splitext(data_path)
            
            if ext.lower() == '.csv':
                self.data = pd.read_csv(data_path)
            elif ext.lower() in ['.xls', '.xlsx']:
                self.data = pd.read_excel(data_path)
            elif ext.lower() == '.json':
                self.data = pd.read_json(data_path)
            else:
                self.logger.error(f"Unsupported file format: {ext}")
                return False
            
            self.logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False
    
    def run_workflow(self, **kwargs):
        """
        Run the complete workflow on the loaded data.
        
        Args:
            **kwargs: Parameters to pass to the workflow
                - target_column (str): Name of the target variable
                - problem_type (str): 'classification' or 'regression'
                - output_format (str): Format of the final report
                - use_llm (bool): Whether to use LLM for enhanced insights
                - llm_provider (str): LLM provider to use ('openai' or 'groq')
                
        Returns:
            tuple: (processed_data, final_report, best_model)
        """
        if self.data is None:
            self.logger.error("No data loaded. Please load data before running the workflow.")
            return None, {"error": "No data loaded"}, None
        
        self.logger.info("Running workflow")
        
        # Add LLM parameters if not explicitly provided
        if 'use_llm' not in kwargs:
            kwargs['use_llm'] = self.use_llm
        
        if 'llm_provider' not in kwargs:
            kwargs['llm_provider'] = self.llm_provider
        
        # Execute workflow
        processed_data, final_report, best_model = self.workflow.execute(
            self.data.copy(), **kwargs
        )
        
        return processed_data, final_report, best_model
    
    def save_report(self, report, output_path):
        """
        Save the generated report to a file.
        
        Args:
            report: The report to save
            output_path (str): Path to save the report
            
        Returns:
            bool: True if report was saved successfully, False otherwise
        """
        self.logger.info(f"Saving report to {output_path}")
        
        try:
            # Determine file extension
            _, ext = os.path.splitext(output_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save report based on format
            if ext.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=4)
            elif ext.lower() in ['.md', '.markdown']:
                with open(output_path, 'w') as f:
                    f.write(report)
            elif ext.lower() == '.html':
                with open(output_path, 'w') as f:
                    f.write(report)
            else:
                self.logger.error(f"Unsupported output format: {ext}")
                return False
            
            self.logger.info(f"Report saved successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            return False
    
    def save_model(self, model, output_path):
        """
        Save the trained model to a file.
        
        Args:
            model: The model to save
            output_path (str): Path to save the model
            
        Returns:
            bool: True if model was saved successfully, False otherwise
        """
        if model is None:
            self.logger.error("No model to save")
            return False
        
        self.logger.info(f"Saving model to {output_path}")
        
        try:
            import joblib
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save model
            joblib.dump(model, output_path)
            
            self.logger.info(f"Model saved successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False 