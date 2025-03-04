"""
Data utility module.
"""
import pandas as pd
import numpy as np
import os

def load_dataset(file_path):
    """
    Load a dataset from a file.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file extension
    _, ext = os.path.splitext(file_path)
    
    # Load data based on file extension
    if ext.lower() == '.csv':
        return pd.read_csv(file_path)
    elif ext.lower() in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif ext.lower() == '.json':
        return pd.read_json(file_path)
    elif ext.lower() == '.parquet':
        return pd.read_parquet(file_path)
    elif ext.lower() == '.pickle' or ext.lower() == '.pkl':
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_dataset(df, file_path):
    """
    Save a dataset to a file.
    
    Args:
        df (pd.DataFrame): Dataset to save
        file_path (str): Path to save the dataset
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Determine file extension
    _, ext = os.path.splitext(file_path)
    
    # Save data based on file extension
    try:
        if ext.lower() == '.csv':
            df.to_csv(file_path, index=False)
        elif ext.lower() == '.xlsx':
            df.to_excel(file_path, index=False)
        elif ext.lower() == '.json':
            df.to_json(file_path, orient='records')
        elif ext.lower() == '.parquet':
            df.to_parquet(file_path, index=False)
        elif ext.lower() == '.pickle' or ext.lower() == '.pkl':
            df.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return True
    except Exception as e:
        print(f"Error saving dataset: {str(e)}")
        return False

def detect_data_types(df):
    """
    Detect and categorize column data types.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary with categorized columns
    """
    data_types = {
        "numerical": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "binary": []
    }
    
    for col in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if binary
            if df[col].nunique() <= 2:
                data_types["binary"].append(col)
            else:
                data_types["numerical"].append(col)
        
        # Check if datetime
        elif pd.api.types.is_datetime64_dtype(df[col]):
            data_types["datetime"].append(col)
        
        # Check if categorical or text
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            # If high cardinality, likely text
            if df[col].nunique() > 20 and df[col].str.len().mean() > 10:
                data_types["text"].append(col)
            # If binary
            elif df[col].nunique() <= 2:
                data_types["binary"].append(col)
            else:
                data_types["categorical"].append(col)
    
    return data_types 