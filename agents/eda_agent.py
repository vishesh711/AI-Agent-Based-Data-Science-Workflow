"""
Exploratory Data Analysis Agent module.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .base_agent import BaseAgent

class EDAAgent(BaseAgent):
    """
    Agent responsible for exploratory data analysis.
    
    Generates statistical summaries, visualizations, and insights.
    """
    
    def __init__(self, name="EDAAgent"):
        """Initialize the EDA Agent."""
        super().__init__(name)
        self.eda_report = {}
    
    def process(self, data, **kwargs):
        """
        Process the data and perform exploratory data analysis.
        
        Args:
            data (pd.DataFrame): Input data (cleaned)
            **kwargs: Additional parameters for EDA
                - target_column (str): Name of the target variable
                - generate_plots (bool): Whether to generate plots
                - correlation_threshold (float): Threshold for correlation analysis
                
        Returns:
            tuple: (data, eda_report)
        """
        self.log_start()
        
        if not self.validate_input(data):
            return data, {"error": "Invalid input data"}
        
        # Extract parameters
        target_column = kwargs.get('target_column')
        generate_plots = kwargs.get('generate_plots', True)
        correlation_threshold = kwargs.get('correlation_threshold', 0.5)
        
        # Initialize EDA report
        self.eda_report = {
            "dataset_info": {},
            "summary_statistics": {},
            "correlation_analysis": {},
            "feature_importance": {},
            "visualizations": {},
            "statistical_tests": {}
        }
        
        # Step 1: Dataset information
        self._analyze_dataset_info(data)
        
        # Step 2: Summary statistics
        self._analyze_summary_statistics(data)
        
        # Step 3: Correlation analysis
        self._analyze_correlations(data, correlation_threshold)
        
        # Step 4: Feature importance (if target column is provided)
        if target_column and target_column in data.columns:
            self._analyze_feature_importance(data, target_column)
        
        # Step 5: Generate visualizations
        if generate_plots:
            self._generate_visualizations(data, target_column)
        
        # Step 6: Statistical tests
        self._perform_statistical_tests(data, target_column)
        
        self.log_end()
        return data, self.eda_report
    
    def _analyze_dataset_info(self, df):
        """
        Analyze basic dataset information.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.logger.info("Analyzing dataset information")
        
        # Get basic information
        self.eda_report["dataset_info"] = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numerical_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
    
    def _analyze_summary_statistics(self, df):
        """
        Analyze summary statistics of the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.logger.info("Analyzing summary statistics")
        
        # Get summary statistics for numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        if not numerical_columns.empty:
            # Convert to dictionary with serializable values
            stats_dict = df[numerical_columns].describe().to_dict()
            
            # Convert numpy values to Python native types
            for col in stats_dict:
                for stat in stats_dict[col]:
                    if isinstance(stats_dict[col][stat], (np.int64, np.float64)):
                        stats_dict[col][stat] = float(stats_dict[col][stat])
            
            self.eda_report["summary_statistics"]["numerical"] = stats_dict
        
        # Get summary statistics for categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        if not categorical_columns.empty:
            cat_stats = {}
            
            for col in categorical_columns:
                value_counts = df[col].value_counts().to_dict()
                
                # Convert keys to strings if they're not already
                value_counts = {str(k): v for k, v in value_counts.items()}
                
                cat_stats[col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": value_counts
                }
            
            self.eda_report["summary_statistics"]["categorical"] = cat_stats
    
    def _analyze_correlations(self, df, threshold):
        """
        Analyze correlations between features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            threshold (float): Correlation threshold
        """
        self.logger.info("Analyzing correlations")
        
        # Get numerical columns for correlation analysis
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_columns) < 2:
            self.logger.info("Not enough numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_columns].corr()
        
        # Convert to dictionary with serializable values
        corr_dict = corr_matrix.to_dict()
        
        # Convert numpy values to Python native types
        for col1 in corr_dict:
            for col2 in corr_dict[col1]:
                if isinstance(corr_dict[col1][col2], (np.int64, np.float64)):
                    corr_dict[col1][col2] = float(corr_dict[col1][col2])
        
        # Find highly correlated features
        high_correlations = []
        
        for i in range(len(numerical_columns)):
            for j in range(i+1, len(numerical_columns)):
                col1 = numerical_columns[i]
                col2 = numerical_columns[j]
                corr = abs(corr_matrix.loc[col1, col2])
                
                if corr >= threshold:
                    high_correlations.append({
                        "feature1": col1,
                        "feature2": col2,
                        "correlation": float(corr)
                    })
        
        # Store results
        self.eda_report["correlation_analysis"] = {
            "correlation_matrix": corr_dict,
            "high_correlations": high_correlations,
            "threshold": threshold
        }
    
    def _analyze_feature_importance(self, df, target_column):
        """
        Analyze feature importance relative to the target.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target variable
        """
        self.logger.info("Analyzing feature importance")
        
        # Check if target column exists
        if target_column not in df.columns:
            self.logger.warning(f"Target column '{target_column}' not found")
            return
        
        # Get numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_columns = [col for col in numerical_columns if col != target_column]
        
        if not numerical_columns:
            self.logger.info("No numerical features for importance analysis")
            return
        
        # Calculate correlation with target
        target_correlations = {}
        
        for col in numerical_columns:
            corr = df[col].corr(df[target_column])
            target_correlations[col] = float(corr)
        
        # Sort by absolute correlation
        sorted_correlations = sorted(
            target_correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Store results
        self.eda_report["feature_importance"] = {
            "target_column": target_column,
            "correlations": dict(sorted_correlations)
        }
    
    def _generate_visualizations(self, df, target_column):
        """
        Generate visualizations for the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target variable
        """
        self.logger.info("Generating visualizations")
        
        # Initialize visualizations dictionary
        visualizations = {
            "distribution_plots": [],
            "correlation_heatmap": None,
            "target_analysis": []
        }
        
        # Get numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Generate distribution plots for numerical columns (limit to top 10)
        for col in numerical_columns[:10]:
            # Create a histogram
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            
            # Save to buffer
            from io import BytesIO
            import base64
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert to base64
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Add to visualizations
            visualizations["distribution_plots"].append({
                "feature": col,
                "plot": plot_data
            })
            
            plt.close()
        
        # Generate correlation heatmap
        if len(numerical_columns) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            
            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert to base64
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Add to visualizations
            visualizations["correlation_heatmap"] = plot_data
            
            plt.close()
        
        # Generate target analysis plots if target column is provided
        if target_column and target_column in df.columns:
            # Check if target is categorical or numerical
            if df[target_column].dtype == 'object' or df[target_column].dtype == 'category' or df[target_column].nunique() < 10:
                # Target is categorical, generate bar plots for top features
                for col in numerical_columns[:5]:
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x=target_column, y=col, data=df)
                    plt.title(f'{col} by {target_column}')
                    plt.tight_layout()
                    
                    # Save to buffer
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    
                    # Convert to base64
                    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
                    
                    # Add to visualizations
                    visualizations["target_analysis"].append({
                        "feature": col,
                        "plot": plot_data
                    })
                    
                    plt.close()
            else:
                # Target is numerical, generate scatter plots for top features
                for col in numerical_columns[:5]:
                    if col != target_column:
                        plt.figure(figsize=(8, 6))
                        sns.scatterplot(x=col, y=target_column, data=df)
                        plt.title(f'{target_column} vs {col}')
                        plt.tight_layout()
                        
                        # Save to buffer
                        buffer = BytesIO()
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        
                        # Convert to base64
                        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
                        
                        # Add to visualizations
                        visualizations["target_analysis"].append({
                            "feature": col,
                            "plot": plot_data
                        })
                        
                        plt.close()
        
        # Store results
        self.eda_report["visualizations"] = visualizations
    
    def _perform_statistical_tests(self, df, target_column):
        """
        Perform statistical tests on the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target variable
        """
        self.logger.info("Performing statistical tests")
        
        # Initialize statistical tests dictionary
        statistical_tests = {
            "normality_tests": {},
            "target_tests": {},
            "suggested": []
        }
        
        # Get numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Perform normality tests on numerical columns
        for col in numerical_columns:
            # Shapiro-Wilk test for normality
            if len(df[col]) < 5000:  # Shapiro-Wilk is only accurate for small samples
                stat, p = stats.shapiro(df[col].dropna())
                
                statistical_tests["normality_tests"][col] = {
                    "test": "shapiro",
                    "statistic": float(stat),
                    "p_value": float(p),
                    "is_normal": p > 0.05
                }
        
        # Perform tests related to the target variable
        if target_column and target_column in df.columns:
            # Check if target is categorical or numerical
            if df[target_column].dtype == 'object' or df[target_column].dtype == 'category' or df[target_column].nunique() < 10:
                # Target is categorical, suggest classification tests
                statistical_tests["suggested"].append("Chi-square test for categorical features")
                statistical_tests["suggested"].append("ANOVA for numerical features across categories")
                
                # Perform ANOVA for numerical features
                for col in numerical_columns:
                    if col != target_column:
                        # Group data by target
                        groups = [df[df[target_column] == val][col].dropna() for val in df[target_column].unique()]
                        
                        # Only perform test if we have enough data
                        if all(len(group) > 0 for group in groups):
                            try:
                                stat, p = stats.f_oneway(*groups)
                                
                                statistical_tests["target_tests"][col] = {
                                    "test": "anova",
                                    "statistic": float(stat),
                                    "p_value": float(p),
                                    "significant": p < 0.05
                                }
                            except:
                                pass
            else:
                # Target is numerical, suggest regression tests
                statistical_tests["suggested"].append("Correlation analysis for numerical features")
                statistical_tests["suggested"].append("T-tests for binary categorical features")
        
        # Store results
        self.eda_report["statistical_tests"] = statistical_tests 