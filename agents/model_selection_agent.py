"""
Model Selection and Hyperparameter Optimization Agent module.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .base_agent import BaseAgent

class ModelSelectionAgent(BaseAgent):
    """
    Agent responsible for model selection and hyperparameter optimization.
    
    Recommends suitable models, tunes hyperparameters, and evaluates performance.
    """
    
    def __init__(self, name="ModelSelectionAgent"):
        """Initialize the Model Selection Agent."""
        super().__init__(name)
        self.model_report = {}
        self.best_model = None
    
    def process(self, data, **kwargs):
        """
        Process the data and select the best model.
        
        Args:
            data (pd.DataFrame): Input data (cleaned)
            **kwargs: Additional parameters for model selection
                - target_column (str): Name of the target variable (required)
                - problem_type (str): 'classification' or 'regression'
                - test_size (float): Size of the test set
                - random_state (int): Random seed for reproducibility
                - cv_folds (int): Number of cross-validation folds
                - scoring (str): Scoring metric for model evaluation
                
        Returns:
            tuple: (data, model_report, best_model)
        """
        self.log_start()
        
        if not self.validate_input(data):
            return data, {"error": "Invalid input data"}, None
        
        # Extract parameters
        target_column = kwargs.get('target_column')
        if not target_column or target_column not in data.columns:
            self.logger.error(f"Target column '{target_column}' not found in data")
            return data, {"error": f"Target column '{target_column}' not found"}, None
        
        problem_type = kwargs.get('problem_type')
        if not problem_type:
            # Auto-detect problem type
            unique_values = data[target_column].nunique()
            if unique_values < 10 or data[target_column].dtype == 'object' or data[target_column].dtype == 'category':
                problem_type = 'classification'
            else:
                problem_type = 'regression'
            self.logger.info(f"Auto-detected problem type: {problem_type}")
        
        test_size = kwargs.get('test_size', 0.2)
        random_state = kwargs.get('random_state', 42)
        cv_folds = kwargs.get('cv_folds', 5)
        
        # Set default scoring based on problem type
        if not kwargs.get('scoring'):
            if problem_type == 'classification':
                scoring = 'f1_weighted'
            else:
                scoring = 'r2'
        else:
            scoring = kwargs.get('scoring')
        
        # Initialize model report
        self.model_report = {
            "problem_type": problem_type,
            "target_column": target_column,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "scoring": scoring,
            "models_evaluated": [],
            "best_model": None
        }
        
        # Split data into features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Evaluate models based on problem type
        if problem_type == 'classification':
            self._evaluate_classification_models(X_train, X_test, y_train, y_test, cv_folds, scoring)
        else:
            self._evaluate_regression_models(X_train, X_test, y_train, y_test, cv_folds, scoring)
        
        self.log_end()
        return data, self.model_report, self.best_model
    
    def _evaluate_classification_models(self, X_train, X_test, y_train, y_test, cv_folds, scoring):
        """
        Evaluate classification models and select the best one.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
            y_train (pd.Series): Training target
            y_test (pd.Series): Testing target
            cv_folds (int): Number of cross-validation folds
            scoring (str): Scoring metric
        """
        # Define models to evaluate
        models = {
            "LogisticRegression": {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["liblinear", "lbfgs"]
                }
            },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                }
            },
            "GradientBoosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                }
            },
            "SVC": {
                "model": SVC(random_state=42),
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                }
            },
            "KNN": {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                }
            },
            "DecisionTree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }
            }
        }
        
        best_score = -float('inf')
        best_model_name = None
        
        # Evaluate each model
        for model_name, model_info in models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            # Perform grid search
            grid_search = GridSearchCV(
                model_info["model"],
                model_info["params"],
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_score = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring).mean()
            
            # Store results
            model_result = {
                "name": model_name,
                "best_params": grid_search.best_params_,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cv_score": cv_score
            }
            
            self.model_report["models_evaluated"].append(model_result)
            
            # Update best model if current is better
            if cv_score > best_score:
                best_score = cv_score
                best_model_name = model_name
                self.best_model = best_model
        
        # Get best model details
        best_model_result = next(
            (model for model in self.model_report["models_evaluated"] if model["name"] == best_model_name),
            None
        )
        
        if best_model_result:
            self.model_report["best_model"] = best_model_result
    
    def _evaluate_regression_models(self, X_train, X_test, y_train, y_test, cv_folds, scoring):
        """
        Evaluate regression models and select the best one.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
            y_train (pd.Series): Training target
            y_test (pd.Series): Testing target
            cv_folds (int): Number of cross-validation folds
            scoring (str): Scoring metric
        """
        # Define models to evaluate
        models = {
            "LinearRegression": {
                "model": LinearRegression(),
                "params": {
                    "fit_intercept": [True, False],
                    "normalize": [True, False]
                }
            },
            "Ridge": {
                "model": Ridge(random_state=42),
                "params": {
                    "alpha": [0.1, 1.0, 10.0],
                    "solver": ["auto", "svd", "cholesky"]
                }
            },
            "Lasso": {
                "model": Lasso(random_state=42),
                "params": {
                    "alpha": [0.1, 1.0, 10.0],
                    "max_iter": [1000, 2000]
                }
            },
            "RandomForest": {
                "model": RandomForestRegressor(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                }
            },
            "GradientBoosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                }
            },
            "SVR": {
                "model": SVR(),
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                }
            },
            "KNN": {
                "model": KNeighborsRegressor(),
                "params": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                }
            }
        }
        
        best_score = -float('inf')
        best_model_name = None
        
        # Evaluate each model
        for model_name, model_info in models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            # Perform grid search
            grid_search = GridSearchCV(
                model_info["model"],
                model_info["params"],
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Cross-validation score
            cv_score = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring).mean()
            
            # Store results
            model_result = {
                "name": model_name,
                "best_params": grid_search.best_params_,
                "r2": r2,
                "mse": mse,
                "rmse": rmse,
                "cv_score": cv_score
            }
            
            self.model_report["models_evaluated"].append(model_result)
            
            # Update best model if current is better
            if cv_score > best_score:
                best_score = cv_score
                best_model_name = model_name
                self.best_model = best_model
        
        # Get best model details
        best_model_result = next(
            (model for model in self.model_report["models_evaluated"] if model["name"] == best_model_name),
            None
        )
        
        if best_model_result:
            self.model_report["best_model"] = best_model_result 