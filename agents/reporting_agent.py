"""
Automated Reporting Agent module.
"""
import pandas as pd
import json
import datetime
from .base_agent import BaseAgent
from utils.llm_api import LLMProvider

class ReportingAgent(BaseAgent):
    """
    Agent responsible for generating human-readable reports.
    
    Summarizes data cleaning steps, EDA insights, and model selection results.
    """
    
    def __init__(self, name="ReportingAgent", use_llm=False, llm_provider="openai"):
        """Initialize the Reporting Agent."""
        super().__init__(name)
        self.use_llm = use_llm
        if use_llm:
            try:
                self.llm = LLMProvider(provider=llm_provider)
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM provider: {str(e)}")
                self.use_llm = False
    
    def process(self, data, cleaning_report=None, eda_report=None, model_report=None, **kwargs):
        """
        Process the reports from other agents and generate a final report.
        
        Args:
            data (pd.DataFrame): Input data
            cleaning_report (dict): Report from Data Cleaning Agent
            eda_report (dict): Report from EDA Agent
            model_report (dict): Report from Model Selection Agent
            **kwargs: Additional parameters for reporting
                - output_format (str): Format of the report ('json', 'html', 'markdown')
                - include_visualizations (bool): Whether to include visualizations
                - use_llm (bool): Whether to include LLM-generated insights
                - llm_provider (str): The LLM provider to use ('openai' or 'groq')
                
        Returns:
            tuple: (data, final_report)
        """
        self.log_start()
        
        # Extract parameters
        output_format = kwargs.get('output_format', 'markdown')
        include_visualizations = kwargs.get('include_visualizations', True)
        use_llm = kwargs.get('use_llm', self.use_llm)
        
        # Initialize final report
        final_report = {
            "title": "Data Science Workflow Report",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": {},
            "data_cleaning": {},
            "exploratory_data_analysis": {},
            "model_selection": {},
            "conclusions": {},
            "llm_insights": {}  # New section for LLM insights
        }
        
        # Add dataset info
        if data is not None:
            final_report["dataset_info"] = {
                "shape": data.shape,
                "columns": list(data.columns),
                "sample": data.head(5).to_dict(orient='records')
            }
        
        # Add data cleaning report
        if cleaning_report:
            final_report["data_cleaning"] = self._summarize_cleaning_report(cleaning_report)
        
        # Add EDA report
        if eda_report:
            final_report["exploratory_data_analysis"] = self._summarize_eda_report(eda_report)
        
        # Add model selection report
        if model_report:
            final_report["model_selection"] = self._summarize_model_report(model_report)
        
        # Generate conclusions
        final_report["conclusions"] = self._generate_conclusions(cleaning_report, eda_report, model_report)
        
        # Add LLM insights if enabled
        if use_llm:
            self._add_llm_insights(final_report, data, cleaning_report, eda_report, model_report)
        
        # Format the report
        formatted_report = self._format_report(final_report, output_format, include_visualizations)
        
        self.log_end()
        return data, formatted_report
    
    def _summarize_cleaning_report(self, cleaning_report):
        """
        Summarize the data cleaning report.
        
        Args:
            cleaning_report (dict): Report from Data Cleaning Agent
            
        Returns:
            dict: Summarized cleaning report
        """
        if not cleaning_report or "error" in cleaning_report:
            return {"status": "No cleaning report available"}
        
        summary = {
            "original_shape": cleaning_report.get("original_shape", None),
            "final_shape": cleaning_report.get("final_shape", None),
            "missing_values_handled": bool(cleaning_report.get("missing_values", {}).get("before", {})),
            "outliers_handled": bool(cleaning_report.get("outliers", {}).get("counts", {})),
            "categorical_encoding": cleaning_report.get("categorical_encoding", {}).get("method", "None"),
            "normalization": cleaning_report.get("normalization", {}).get("method", "None")
        }
        
        return summary
    
    def _summarize_eda_report(self, eda_report):
        """Summarize the EDA report for the final report."""
        if not eda_report or not isinstance(eda_report, dict):
            return {"summary": "No EDA report available"}
        
        summary = {}
        
        # Extract dataset info
        if "dataset_info" in eda_report:
            info = eda_report["dataset_info"]
            summary["dataset_shape"] = info.get("shape", "N/A")
            summary["numerical_features"] = len(info.get("numerical_columns", []))
            summary["categorical_features"] = len(info.get("categorical_columns", []))
        
        # Extract correlation info
        if "correlations" in eda_report:
            corr = eda_report["correlations"]
            if isinstance(corr, dict) and "strong_correlations" in corr:
                summary["strong_correlations"] = len(corr["strong_correlations"])
            else:
                summary["strong_correlations"] = 0
        
        # Extract statistical tests
        if "statistical_tests" in eda_report:
            tests = eda_report["statistical_tests"]
            if isinstance(tests, list):
                # Fix: Handle both string and dictionary items in the list
                suggested_tests = []
                for test in tests:
                    if isinstance(test, dict) and "test" in test:
                        suggested_tests.append(test["test"])
                    elif isinstance(test, str):
                        suggested_tests.append(test)
                summary["suggested_statistical_tests"] = suggested_tests
            else:
                summary["suggested_statistical_tests"] = []
        
        # Extract visualizations
        if "visualizations" in eda_report:
            viz = eda_report["visualizations"]
            if isinstance(viz, list):
                summary["visualizations_count"] = len(viz)
            else:
                summary["visualizations_count"] = 0
        
        return summary
    
    def _summarize_model_report(self, model_report):
        """
        Summarize the model selection report.
        
        Args:
            model_report (dict): Report from Model Selection Agent
            
        Returns:
            dict: Summarized model report
        """
        if not model_report or "error" in model_report:
            return {"status": "No model selection report available"}
        
        # Extract problem type
        problem_type = model_report.get("problem_type", "Unknown")
        
        # Extract best model
        best_model = model_report.get("best_model", {})
        
        # Extract model comparison
        models_evaluated = model_report.get("models_evaluated", [])
        model_comparison = []
        
        if problem_type == "classification":
            for model in models_evaluated:
                model_comparison.append({
                    "name": model.get("name", "Unknown"),
                    "accuracy": model.get("accuracy", 0),
                    "f1": model.get("f1", 0)
                })
        else:  # regression
            for model in models_evaluated:
                model_comparison.append({
                    "name": model.get("name", "Unknown"),
                    "r2": model.get("r2", 0),
                    "rmse": model.get("rmse", 0)
                })
        
        summary = {
            "problem_type": problem_type,
            "target_column": model_report.get("target_column", "Unknown"),
            "models_evaluated": len(models_evaluated),
            "best_model": {
                "name": best_model.get("name", "Unknown"),
                "params": best_model.get("best_params", {}),
                "performance": {
                    "accuracy": best_model.get("accuracy", None),
                    "f1": best_model.get("f1", None),
                    "r2": best_model.get("r2", None),
                    "rmse": best_model.get("rmse", None)
                }
            },
            "model_comparison": model_comparison
        }
        
        return summary
    
    def _generate_conclusions(self, cleaning_report, eda_report, model_report):
        """
        Generate conclusions based on the reports from other agents.
        
        Args:
            cleaning_report (dict): Report from Data Cleaning Agent
            eda_report (dict): Report from EDA Agent
            model_report (dict): Report from Model Selection Agent
            
        Returns:
            dict: Conclusions
        """
        conclusions = {
            "data_quality": "Unknown",
            "key_findings": [],
            "model_recommendations": [],
            "next_steps": []
        }
        
        # Assess data quality
        if cleaning_report and "error" not in cleaning_report:
            # Check if there were significant changes in shape
            original_shape = cleaning_report.get("original_shape")
            final_shape = cleaning_report.get("final_shape")
            
            if original_shape and final_shape:
                original_rows = original_shape[0] if isinstance(original_shape, tuple) else 0
                final_rows = final_shape[0] if isinstance(final_shape, tuple) else 0
                
                if original_rows > 0:
                    data_loss_percentage = (original_rows - final_rows) / original_rows * 100
                    
                    if data_loss_percentage > 30:
                        conclusions["data_quality"] = "Poor"
                        conclusions["key_findings"].append(f"Significant data loss during cleaning ({data_loss_percentage:.1f}%)")
                    elif data_loss_percentage > 10:
                        conclusions["data_quality"] = "Fair"
                        conclusions["key_findings"].append(f"Moderate data loss during cleaning ({data_loss_percentage:.1f}%)")
                    else:
                        conclusions["data_quality"] = "Good"
            
            # Check for missing values and outliers
            missing_values_handled = cleaning_report.get("missing_values", {}).get("before", {})
            outliers_handled = cleaning_report.get("outliers", {}).get("counts", {})
            
            if missing_values_handled:
                conclusions["key_findings"].append("Missing values were present and handled")
            
            if outliers_handled:
                conclusions["key_findings"].append("Outliers were detected and handled")
        
        # Extract insights from EDA
        if eda_report and "error" not in eda_report:
            # Check for strong correlations
            correlations = eda_report.get("correlation_analysis", {}).get("strong_correlations", [])
            
            if correlations:
                conclusions["key_findings"].append(f"Found {len(correlations)} strong feature correlations")
                
                # Add top correlations
                for i, corr in enumerate(correlations[:3]):
                    conclusions["key_findings"].append(
                        f"Strong correlation between {corr.get('feature1')} and {corr.get('feature2')}: {corr.get('correlation', 0):.2f}"
                    )
            
            # Check for non-normal distributions
            non_normal_features = []
            for col, analysis in eda_report.get("numerical_analysis", {}).items():
                if isinstance(analysis, dict) and not analysis.get("normality_test", {}).get("is_normal", True):
                    non_normal_features.append(col)
            
            if non_normal_features:
                conclusions["key_findings"].append(f"{len(non_normal_features)} features have non-normal distributions")
        
        # Extract model insights
        if model_report and "error" not in model_report:
            problem_type = model_report.get("problem_type", "Unknown")
            best_model = model_report.get("best_model", {})
            
            if best_model:
                model_name = best_model.get("name", "Unknown")
                
                conclusions["key_findings"].append(f"Best performing model: {model_name}")
                
                # Add performance metrics
                if problem_type == "classification":
                    accuracy = best_model.get("accuracy")
                    f1 = best_model.get("f1")
                    
                    if accuracy is not None:
                        conclusions["key_findings"].append(f"Model accuracy: {accuracy:.2f}")
                    
                    if f1 is not None:
                        conclusions["key_findings"].append(f"Model F1 score: {f1:.2f}")
                else:  # regression
                    r2 = best_model.get("r2")
                    rmse = best_model.get("rmse")
                    
                    if r2 is not None:
                        conclusions["key_findings"].append(f"Model R² score: {r2:.2f}")
                    
                    if rmse is not None:
                        conclusions["key_findings"].append(f"Model RMSE: {rmse:.2f}")
            
            # Generate model recommendations
            if best_model:
                conclusions["model_recommendations"].append(f"Use {best_model.get('name', 'the best model')} for predictions")
                
                # Add hyperparameter recommendations
                best_params = best_model.get("best_params", {})
                if best_params:
                    param_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
                    conclusions["model_recommendations"].append(f"Optimal hyperparameters: {param_str}")
            
            # Compare with other models
            models_evaluated = model_report.get("models_evaluated", [])
            if len(models_evaluated) > 1:
                conclusions["model_recommendations"].append(f"Evaluated {len(models_evaluated)} different models")
        
        # Generate next steps
        conclusions["next_steps"] = [
            "Validate the model on new data",
            "Consider feature engineering to improve model performance",
            "Implement the model in a production environment",
            "Monitor model performance over time"
        ]
        
        # Add data-specific next steps
        if conclusions["data_quality"] == "Poor":
            conclusions["next_steps"].insert(0, "Collect more high-quality data")
            conclusions["next_steps"].insert(0, "Investigate and address data quality issues")
        
        return conclusions
    
    def _add_llm_insights(self, report, data, cleaning_report, eda_report, model_report):
        """
        Add LLM-generated insights to the report.
        
        Args:
            report (dict): The report to add insights to
            data (pd.DataFrame): The dataset
            cleaning_report (dict): Report from Data Cleaning Agent
            eda_report (dict): Report from EDA Agent
            model_report (dict): Report from Model Selection Agent
        """
        self.logger.info("Generating LLM insights")
        
        try:
            # Create data description for LLM
            data_description = self._create_data_description(data, eda_report)
            
            # Generate general insights
            report["llm_insights"]["general_insights"] = self.llm.generate_insights(data_description)
            
            # Generate model explanation if available
            if model_report and "best_model" in model_report and model_report["best_model"]:
                model_info = json.dumps(model_report["best_model"], indent=2)
                data_info = json.dumps({"shape": data.shape, "columns": list(data.columns)}, indent=2)
                report["llm_insights"]["model_explanation"] = self.llm.explain_model_results(model_info, data_info)
            
            # Generate next steps recommendations
            workflow_results = {
                "data_cleaning": self._summarize_cleaning_report(cleaning_report),
                "eda": self._summarize_eda_report(eda_report),
                "model": self._summarize_model_report(model_report)
            }
            report["llm_insights"]["next_steps"] = self.llm.suggest_next_steps(json.dumps(workflow_results, indent=2))
            
        except Exception as e:
            self.logger.error(f"Error generating LLM insights: {str(e)}")
            report["llm_insights"]["error"] = f"Failed to generate insights: {str(e)}"
    
    def _create_data_description(self, data, eda_report):
        """Create a textual description of the data for the LLM."""
        description = f"Dataset with {data.shape[0]} rows and {data.shape[1]} columns.\n\n"
        
        # Add column information
        description += "Columns:\n"
        for col in data.columns:
            dtype = str(data[col].dtype)
            unique = data[col].nunique()
            missing = data[col].isna().sum()
            description += f"- {col} (type: {dtype}, unique values: {unique}, missing: {missing})\n"
        
        # Add correlation information if available
        if eda_report and "correlation_analysis" in eda_report and "strong_correlations" in eda_report["correlation_analysis"]:
            description += "\nStrong correlations:\n"
            for corr in eda_report["correlation_analysis"]["strong_correlations"]:
                description += f"- {corr.get('feature1', '')} and {corr.get('feature2', '')}: {corr.get('correlation', 0):.2f}\n"
        
        return description
    
    def _format_report(self, report, output_format, include_visualizations):
        """
        Format the final report in the specified format.
        
        Args:
            report (dict): Final report
            output_format (str): Format of the report ('json', 'html', 'markdown')
            include_visualizations (bool): Whether to include visualizations
            
        Returns:
            str: Formatted report
        """
        if output_format == 'json':
            return json.dumps(report, indent=2)
        
        elif output_format == 'html':
            # In a real implementation, this would generate HTML
            # For this example, we'll return a simple HTML structure
            html = f"""
            <html>
            <head>
                <title>{report['title']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .section {{ margin-bottom: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>{report['title']}</h1>
                <p>Generated on: {report['timestamp']}</p>
                
                <div class="section">
                    <h2>Dataset Information</h2>
                    <p>Shape: {report['dataset_info'].get('shape', 'N/A')}</p>
                    <p>Columns: {', '.join(report['dataset_info'].get('columns', []))}</p>
                </div>
                
                <div class="section">
                    <h2>Data Cleaning Summary</h2>
                    <p>Original Shape: {report['data_cleaning'].get('original_shape', 'N/A')}</p>
                    <p>Final Shape: {report['data_cleaning'].get('final_shape', 'N/A')}</p>
                    <p>Missing Values Handled: {report['data_cleaning'].get('missing_values_handled', 'No')}</p>
                    <p>Outliers Handled: {report['data_cleaning'].get('outliers_handled', 'No')}</p>
                </div>
                
                <div class="section">
                    <h2>Model Selection Summary</h2>
                    <p>Problem Type: {report['model_selection'].get('problem_type', 'N/A')}</p>
                    <p>Best Model: {report['model_selection'].get('best_model', {}).get('name', 'N/A')}</p>
                </div>
                
                <div class="section">
                    <h2>Conclusions</h2>
                    <p>Data Quality: {report['conclusions'].get('data_quality', 'N/A')}</p>
                    <h3>Key Findings</h3>
                    <ul>
                        {''.join([f'<li>{finding}</li>' for finding in report['conclusions'].get('key_findings', [])])}
                    </ul>
                    <h3>Model Recommendations</h3>
                    <ul>
                        {''.join([f'<li>{rec}</li>' for rec in report['conclusions'].get('model_recommendations', [])])}
                    </ul>
                    <h3>Next Steps</h3>
                    <ul>
                        {''.join([f'<li>{step}</li>' for step in report['conclusions'].get('next_steps', [])])}
                    </ul>
                </div>
            </body>
            </html>
            """
            return html
        
        else:  # markdown (default)
            markdown = f"""
            # {report['title']}
            
            Generated on: {report['timestamp']}
            
            ## Dataset Information
            
            - Shape: {report['dataset_info'].get('shape', 'N/A')}
            - Columns: {', '.join(report['dataset_info'].get('columns', []))}
            
            ## Data Cleaning Summary
            
            - Original Shape: {report['data_cleaning'].get('original_shape', 'N/A')}
            - Final Shape: {report['data_cleaning'].get('final_shape', 'N/A')}
            - Missing Values Handled: {report['data_cleaning'].get('missing_values_handled', 'No')}
            - Outliers Handled: {report['data_cleaning'].get('outliers_handled', 'No')}
            - Categorical Encoding: {report['data_cleaning'].get('categorical_encoding', 'None')}
            - Normalization: {report['data_cleaning'].get('normalization', 'None')}
            
            ## Exploratory Data Analysis Summary
            
            - Numerical Features: {report['exploratory_data_analysis'].get('numerical_features', 0)}
            - Categorical Features: {report['exploratory_data_analysis'].get('categorical_features', 0)}
            
            ### Strong Correlations
            
            {self._format_correlations_markdown(report['exploratory_data_analysis'].get('strong_correlations', []))}
            
            ## Model Selection Summary
            
            - Problem Type: {report['model_selection'].get('problem_type', 'N/A')}
            - Target Column: {report['model_selection'].get('target_column', 'N/A')}
            - Models Evaluated: {report['model_selection'].get('models_evaluated', 0)}
            
            ### Best Model
            
            - Name: {report['model_selection'].get('best_model', {}).get('name', 'N/A')}
            - Parameters: {report['model_selection'].get('best_model', {}).get('params', {})}
            - Performance: {self._format_performance_markdown(report['model_selection'].get('best_model', {}).get('performance', {}))}
            
            ### Model Comparison
            
            {self._format_model_comparison_markdown(report['model_selection'].get('model_comparison', []), report['model_selection'].get('problem_type', 'Unknown'))}
            
            ## Conclusions
            
            - Data Quality: {report['conclusions'].get('data_quality', 'N/A')}
            
            ### Key Findings
            
            {self._format_list_markdown(report['conclusions'].get('key_findings', []))}
            
            ### Model Recommendations
            
            {self._format_list_markdown(report['conclusions'].get('model_recommendations', []))}
            
            ### Next Steps
            
            {self._format_list_markdown(report['conclusions'].get('next_steps', []))}
            """
            
            # Add LLM insights section
            if "llm_insights" in report and report["llm_insights"]:
                markdown += "\n## AI-Generated Insights\n\n"
                
                if "general_insights" in report["llm_insights"]:
                    markdown += "### General Insights\n\n"
                    markdown += report["llm_insights"]["general_insights"] + "\n\n"
                
                if "model_explanation" in report["llm_insights"]:
                    markdown += "### Model Explanation\n\n"
                    markdown += report["llm_insights"]["model_explanation"] + "\n\n"
                
                if "next_steps" in report["llm_insights"]:
                    markdown += "### Recommended Next Steps\n\n"
                    markdown += report["llm_insights"]["next_steps"] + "\n\n"
            
            return markdown
    
    def _format_correlations_markdown(self, correlations):
        """Format correlations as markdown."""
        if not correlations:
            return "No strong correlations found."
        
        markdown = ""
        for corr in correlations:
            markdown += f"- {corr.get('feature1', '')} and {corr.get('feature2', '')}: {corr.get('correlation', 0):.2f}\n"
        
        return markdown
    
    def _format_performance_markdown(self, performance):
        """Format model performance as markdown."""
        metrics = []
        
        if performance.get('accuracy') is not None:
            metrics.append(f"Accuracy: {performance.get('accuracy', 0):.4f}")
        
        if performance.get('f1') is not None:
            metrics.append(f"F1 Score: {performance.get('f1', 0):.4f}")
        
        if performance.get('r2') is not None:
            metrics.append(f"R² Score: {performance.get('r2', 0):.4f}")
        
        if performance.get('rmse') is not None:
            metrics.append(f"RMSE: {performance.get('rmse', 0):.4f}")
        
        return ", ".join(metrics)
    
    def _format_model_comparison_markdown(self, models, problem_type):
        """Format model comparison as markdown."""
        if not models:
            return "No model comparison available."
        
        markdown = "| Model | "
        
        if problem_type == "classification":
            markdown += "Accuracy | F1 Score |\n| --- | --- | --- |\n"
            
            for model in models:
                markdown += f"| {model.get('name', 'Unknown')} | {model.get('accuracy', 0):.4f} | {model.get('f1', 0):.4f} |\n"
        else:
            markdown += "R² Score | RMSE |\n| --- | --- | --- |\n"
            
            for model in models:
                markdown += f"| {model.get('name', 'Unknown')} | {model.get('r2', 0):.4f} | {model.get('rmse', 0):.4f} |\n"
        
        return markdown
    
    def _format_list_markdown(self, items):
        """Format a list as markdown."""
        if not items:
            return "None."
        
        markdown = ""
        for item in items:
            markdown += f"- {item}\n"
        
        return markdown 