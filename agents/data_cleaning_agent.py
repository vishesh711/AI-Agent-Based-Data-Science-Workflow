"""
Data Cleaning Agent module.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from .base_agent import BaseAgent

class DataCleaningAgent(BaseAgent):
    """
    Agent responsible for data cleaning and preprocessing.
    
    Handles missing values, outliers, normalization, and encoding.
    """
    
    def __init__(self, name="DataCleaningAgent"):
        """Initialize the Data Cleaning Agent."""
        super().__init__(name)
    
    def process(self, data, **kwargs):
        """
        Process the data by cleaning and preprocessing.
        
        Args:
            data (pd.DataFrame): Input data
            **kwargs: Additional parameters for data cleaning
                - handle_missing (bool): Whether to handle missing values
                - handle_outliers (bool): Whether to handle outliers
                - normalize (bool): Whether to normalize numerical features
                - encode_categorical (bool): Whether to encode categorical features
                
        Returns:
            tuple: (cleaned_data, cleaning_report)
        """
        self.log_start()
        
        if not self.validate_input(data):
            return None, {"error": "Invalid input data"}
        
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Initialize cleaning report
        cleaning_report = {
            "original_shape": df.shape,
            "missing_values": {},
            "outliers": {},
            "normalization": {},
            "encoding": {},
            "final_shape": None
        }
        
        # Extract parameters
        handle_missing = kwargs.get('handle_missing', True)
        handle_outliers = kwargs.get('handle_outliers', True)
        normalize = kwargs.get('normalize', True)
        encode_categorical = kwargs.get('encode_categorical', True)
        
        # Step 1: Handle missing values
        if handle_missing:
            df, missing_report = self._handle_missing_values(df)
            cleaning_report["missing_values"] = missing_report
        
        # Step 2: Handle outliers
        if handle_outliers:
            df, outliers_report = self._handle_outliers(df)
            cleaning_report["outliers"] = outliers_report
        
        # Step 3: Normalize numerical features
        if normalize:
            df, normalization_report = self._normalize_features(df)
            cleaning_report["normalization"] = normalization_report
        
        # Step 4: Encode categorical features
        if encode_categorical:
            df, encoding_report = self._encode_categorical(df)
            cleaning_report["encoding"] = encoding_report
        
        # Update final shape
        cleaning_report["final_shape"] = df.shape
        
        self.log_end()
        return df, cleaning_report
    
    def _handle_missing_values(self, df):
        """
        Handle missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (processed_df, missing_report)
        """
        self.logger.info("Handling missing values")
        
        # Initialize report
        missing_report = {
            "total_missing": df.isna().sum().sum(),
            "columns_with_missing": {},
            "strategy": {}
        }
        
        # Check for missing values
        missing_values = df.isna().sum()
        columns_with_missing = missing_values[missing_values > 0]
        
        if columns_with_missing.empty:
            self.logger.info("No missing values found")
            return df, missing_report
        
        # Record columns with missing values
        for column, count in columns_with_missing.items():
            missing_report["columns_with_missing"][column] = int(count)
        
        # Handle missing values based on data type
        for column in columns_with_missing.index:
            if pd.api.types.is_numeric_dtype(df[column]):
                # For numerical columns, use mean imputation
                imputer = SimpleImputer(strategy='mean')
                df[column] = imputer.fit_transform(df[[column]])
                missing_report["strategy"][column] = "mean_imputation"
            else:
                # For categorical columns, use most frequent imputation
                imputer = SimpleImputer(strategy='most_frequent')
                df[column] = imputer.fit_transform(df[[column]])
                missing_report["strategy"][column] = "most_frequent_imputation"
        
        return df, missing_report
    
    def _handle_outliers(self, df):
        """
        Handle outliers in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (processed_df, outliers_report)
        """
        self.logger.info("Handling outliers")
        
        # Initialize report
        outliers_report = {
            "columns_with_outliers": {},
            "strategy": {}
        }
        
        # Only process numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        for column in numerical_columns:
            # Calculate IQR
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outliers_report["columns_with_outliers"][column] = int(outlier_count)
                
                # Cap outliers at the bounds
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                outliers_report["strategy"][column] = "capping"
        
        return df, outliers_report
    
    def _normalize_features(self, df):
        """
        Normalize numerical features in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (processed_df, normalization_report)
        """
        self.logger.info("Normalizing features")
        
        # Initialize report
        normalization_report = {
            "normalized_columns": [],
            "strategy": "standard_scaling"
        }
        
        # Only normalize numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_columns) > 0:
            # Use StandardScaler for normalization
            scaler = StandardScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            
            # Record normalized columns
            normalization_report["normalized_columns"] = numerical_columns.tolist()
        
        return df, normalization_report
    
    def _encode_categorical(self, df):
        """
        Encode categorical features in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (processed_df, encoding_report)
        """
        self.logger.info("Encoding categorical features")
        
        # Initialize report
        encoding_report = {
            "encoded_columns": {},
            "strategy": {}
        }
        
        # Only encode object and category columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            # Check cardinality
            unique_values = df[column].nunique()
            
            if unique_values <= 10:  # Low cardinality, use one-hot encoding
                # Create a temporary dataframe with one-hot encoded columns
                encoder = OneHotEncoder(sparse=False, drop='first')
                encoded = encoder.fit_transform(df[[column]])
                
                # Create new column names
                categories = encoder.categories_[0][1:]  # Skip the first category (dropped)
                new_columns = [f"{column}_{cat}" for cat in categories]
                
                # Add encoded columns to the dataframe
                for i, new_col in enumerate(new_columns):
                    df[new_col] = encoded[:, i]
                
                # Drop the original column
                df = df.drop(column, axis=1)
                
                # Record encoding
                encoding_report["encoded_columns"][column] = new_columns
                encoding_report["strategy"][column] = "one_hot_encoding"
            else:  # High cardinality, use label encoding
                # Convert to category codes
                df[column] = df[column].astype('category').cat.codes
                
                # Record encoding
                encoding_report["encoded_columns"][column] = [column]
                encoding_report["strategy"][column] = "label_encoding"
        
        return df, encoding_report 