"""
Workflow Execution module.
"""
import logging
from agents.data_cleaning_agent import DataCleaningAgent
from agents.eda_agent import EDAAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.reporting_agent import ReportingAgent

class Workflow:
    """
    Manages the execution order of the AI agents.
    
    Ensures proper data flow between agents and handles dependencies.
    """
    
    def __init__(self, use_llm=False, llm_provider="openai"):
        """
        Initialize the workflow.
        
        Args:
            use_llm (bool): Whether to use LLM for enhanced insights
            llm_provider (str): LLM provider to use ('openai' or 'groq')
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize agents
        self.data_cleaning_agent = DataCleaningAgent()
        self.eda_agent = EDAAgent()
        self.model_selection_agent = ModelSelectionAgent()
        self.reporting_agent = ReportingAgent(use_llm=use_llm, llm_provider=llm_provider)
        
        # Initialize reports
        self.cleaning_report = None
        self.eda_report = None
        self.model_report = None
        self.best_model = None
        self.final_report = None
        
        # LLM settings
        self.use_llm = use_llm
        self.llm_provider = llm_provider
    
    def execute(self, data, **kwargs):
        """
        Execute the workflow on the given data.
        
        Args:
            data (pd.DataFrame): Input data
            **kwargs: Additional parameters for the workflow
                - target_column (str): Name of the target variable
                - problem_type (str): 'classification' or 'regression'
                - output_format (str): Format of the final report
                
        Returns:
            tuple: (processed_data, final_report, best_model)
        """
        self.logger.info("Starting workflow execution")
        
        # Extract parameters
        target_column = kwargs.get('target_column')
        problem_type = kwargs.get('problem_type')
        output_format = kwargs.get('output_format', 'markdown')
        
        # Extract LLM parameters
        use_llm = kwargs.get('use_llm', self.use_llm)
        llm_provider = kwargs.get('llm_provider', self.llm_provider)
        
        # Step 1: Data Cleaning
        self.logger.info("Step 1: Data Cleaning")
        cleaned_data, self.cleaning_report = self.data_cleaning_agent.process(
            data,
            handle_missing=kwargs.get('handle_missing', True),
            handle_outliers=kwargs.get('handle_outliers', True),
            normalize=kwargs.get('normalize', True),
            encode_categorical=kwargs.get('encode_categorical', True)
        )
        
        if cleaned_data is None:
            self.logger.error("Data cleaning failed")
            return None, {"error": "Data cleaning failed"}, None
        
        # Step 2: Exploratory Data Analysis
        self.logger.info("Step 2: Exploratory Data Analysis")
        _, self.eda_report = self.eda_agent.process(
            cleaned_data,
            target_column=target_column,
            generate_plots=kwargs.get('generate_plots', True),
            correlation_threshold=kwargs.get('correlation_threshold', 0.5)
        )
        
        # Step 3: Model Selection
        self.logger.info("Step 3: Model Selection")
        if target_column and target_column in cleaned_data.columns:
            _, self.model_report, self.best_model = self.model_selection_agent.process(
                cleaned_data,
                target_column=target_column,
                problem_type=problem_type,
                test_size=kwargs.get('test_size', 0.2),
                random_state=kwargs.get('random_state', 42),
                cv_folds=kwargs.get('cv_folds', 5),
                scoring=kwargs.get('scoring')
            )
        else:
            self.logger.warning(f"Target column '{target_column}' not found or not specified. Skipping model selection.")
        
        # Step 4: Reporting
        self.logger.info("Step 4: Generating Report")
        _, self.final_report = self.reporting_agent.process(
            cleaned_data,
            cleaning_report=self.cleaning_report,
            eda_report=self.eda_report,
            model_report=self.model_report,
            output_format=output_format,
            include_visualizations=kwargs.get('include_visualizations', True),
            use_llm=use_llm,
            llm_provider=llm_provider
        )
        
        self.logger.info("Workflow execution completed")
        return cleaned_data, self.final_report, self.best_model 