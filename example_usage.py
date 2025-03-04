from main_orchestrator import MainOrchestrator
import os
import pandas as pd
import numpy as np

# Set API keys as environment variables
os.environ["GROQ_API_KEY"] = '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'

# Load the dataset
data_path = "/Users/vishesh/Documents/Github/AI-Agent-Based-Data-Science-Workflow/customers-1000.csv"
df = pd.read_csv(data_path)

# Print available columns to help identify a suitable target
print("Available columns in the dataset:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# Create a synthetic target column for demonstration purposes
if 'Subscription Date' in df.columns:
    # Convert to datetime first
    df['Subscription Date'] = pd.to_datetime(df['Subscription Date'], errors='coerce')
    
    # Extract useful features from the datetime
    df['Sub_Month'] = df['Subscription Date'].dt.month
    df['Sub_Year'] = df['Subscription Date'].dt.year
    df['Sub_Day'] = df['Subscription Date'].dt.day
    df['Sub_DayOfWeek'] = df['Subscription Date'].dt.dayofweek
    
    # Create target column (1 if subscribed in second half of year, 0 otherwise)
    df['target'] = df['Sub_Month'].apply(lambda x: 1 if x > 6 else 0).astype(int)
    
    # Drop the original datetime column to avoid type conflicts
    df = df.drop('Subscription Date', axis=1)
    
    print("\nCreated synthetic 'target' column based on subscription month")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"Target dtype: {df['target'].dtype}")
    print(f"Added datetime features: Sub_Month, Sub_Year, Sub_Day, Sub_DayOfWeek")

# Initialize with LLM capabilities enabled
orchestrator = MainOrchestrator(
    data=df,  # Pass the DataFrame directly instead of the path
    use_llm=True,
    llm_provider="groq"
)

# Run the workflow with LLM insights
try:
    # Create a custom function to safely handle the report
    def safe_get(obj, key, default=None):
        """Safely get a value from a dictionary or return default."""
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        return default
    
    cleaned_data, report, best_model = orchestrator.run_workflow(
        target_column="target",
        problem_type="classification",  # Explicitly set as classification
        output_format="markdown",
        use_llm=True,
        llm_provider="groq",
        # Add parameters to simplify the workflow for testing
        handle_missing=True,
        handle_outliers=False,  # Skip outlier handling to speed up
        normalize=False,        # Skip normalization to avoid issues
        encode_categorical=True
    )

    # Print a summary of the report
    print("\nWorkflow completed!")
    print(f"Data shape after processing: {cleaned_data.shape if cleaned_data is not None else 'N/A'}")
    
    # Safely access best model information
    if isinstance(best_model, dict) and 'name' in best_model:
        print(f"Best model: {best_model['name']}")
    else:
        print("Best model: None or not in expected format")

    # Save the report to a file with better formatting
    with open("data_science_report.md", "w") as f:
        # Handle different report formats
        if isinstance(report, dict) and "markdown_report" in report:
            f.write(report["markdown_report"])
        elif isinstance(report, dict):
            # Convert the report dictionary to markdown
            # Create a more readable markdown report
            md_report = f"# Data Science Workflow Report\n\n"
            md_report += f"## Dataset Information\n\n"
            
            # Safely access dataset info
            info = safe_get(report, "dataset_info", {})
            md_report += f"- Shape: {safe_get(info, 'shape', 'N/A')}\n"
            md_report += f"- Columns: {', '.join(safe_get(info, 'columns', []))}\n\n"
            
            # Safely access data cleaning info
            if "data_cleaning" in report:
                md_report += f"## Data Cleaning Summary\n\n"
                cleaning = report["data_cleaning"]
                if isinstance(cleaning, dict):
                    for key, value in cleaning.items():
                        md_report += f"- {key}: {value}\n"
                else:
                    md_report += f"- Summary: {cleaning}\n"
                md_report += "\n"
            
            # Safely access EDA info
            if "exploratory_data_analysis" in report:
                md_report += f"## Exploratory Data Analysis\n\n"
                eda = report["exploratory_data_analysis"]
                if isinstance(eda, dict):
                    for key, value in eda.items():
                        if isinstance(value, dict):
                            md_report += f"### {key}\n\n"
                            for subkey, subvalue in value.items():
                                md_report += f"- {subkey}: {subvalue}\n"
                        else:
                            md_report += f"- {key}: {value}\n"
                else:
                    md_report += f"- Summary: {eda}\n"
                md_report += "\n"
            
            # Safely access model selection info
            if "model_selection" in report:
                md_report += f"## Model Selection\n\n"
                model = report["model_selection"]
                if isinstance(model, dict):
                    for key, value in model.items():
                        md_report += f"- {key}: {value}\n"
                else:
                    md_report += f"- Summary: {model}\n"
                md_report += "\n"
            
            # Safely access LLM insights
            if "llm_insights" in report:
                md_report += f"## AI-Generated Insights\n\n"
                insights = report["llm_insights"]
                if isinstance(insights, dict):
                    for key, value in insights.items():
                        md_report += f"### {key}\n\n{value}\n\n"
                else:
                    md_report += f"{insights}\n\n"
            
            f.write(md_report)
        else:
            # If it's a string or other type
            f.write(str(report))
        
        print("\nEnhanced report saved to data_science_report.md")

except Exception as e:
    print(f"\nError in workflow: {str(e)}")
    import traceback
    traceback.print_exc() 