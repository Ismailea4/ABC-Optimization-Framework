"""
SVM hyperparameter optimization using ABC algorithms
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from .custom_problem import OptimizationProblem


class SVMOptimizationProblem(OptimizationProblem):
    """
    SVM hyperparameter optimization problem for ABC algorithms.
    """
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 problem_type: str = 'classification',
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 scoring: str = 'auto',
                 kernel_type: str = 'rbf',
                 optimize_params: List[str] = None):
        """
        Initialize SVM optimization problem.
        
        Args:
            X: Feature matrix
            y: Target vector
            problem_type: 'classification' or 'regression'
            cv_folds: Number of cross-validation folds
            test_size: Test set proportion for final evaluation
            random_state: Random state for reproducibility
            scoring: Scoring metric ('auto', 'accuracy', 'f1', 'mse', etc.)
            kernel_type: SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            optimize_params: List of parameters to optimize
        """
        # Define parameter bounds based on kernel type and parameters to optimize
        if optimize_params is None:
            if kernel_type == 'rbf':
                optimize_params = ['C', 'gamma']
            elif kernel_type == 'poly':
                optimize_params = ['C', 'gamma', 'degree']
            elif kernel_type == 'linear':
                optimize_params = ['C']
            else:  # sigmoid
                optimize_params = ['C', 'gamma']
        
        self.optimize_params = optimize_params
        bounds = self._get_parameter_bounds(optimize_params)
        dimensions = len(bounds)
        
        super().__init__(dimensions, bounds)
        
        self.X = X
        self.y = y
        self.problem_type = problem_type.lower()
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.kernel_type = kernel_type
        
        # Split data for final evaluation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if problem_type == 'classification' else None
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Set up cross-validation
        if self.problem_type == 'classification':
            self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Set scoring metric
        self.scoring = self._get_scoring_metric(scoring)
        
        # Track evaluations
        self.evaluation_count = 0
        self.best_params = None
        self.best_score = float('-inf') if self._is_score_better_when_higher() else float('inf')
        
    def _get_parameter_bounds(self, optimize_params: List[str]) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds_dict = {
            'C': (0.001, 1000.0),           # Regularization parameter (log scale)
            'gamma': (1e-6, 10.0),          # Kernel coefficient (log scale)
            'degree': (1, 10),              # Polynomial degree
            'coef0': (-10.0, 10.0),         # Independent term in kernel
            'epsilon': (0.001, 1.0),        # Epsilon for SVR
        }
        
        return [bounds_dict[param] for param in optimize_params]
    
    def _get_scoring_metric(self, scoring: str) -> str:
        """Get appropriate scoring metric."""
        if scoring == 'auto':
            if self.problem_type == 'classification':
                return 'accuracy'
            else:
                return 'neg_mean_squared_error'
        return scoring
    
    def _is_score_better_when_higher(self) -> bool:
        """Check if higher score is better for the current metric."""
        higher_is_better = ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
                           'precision', 'recall', 'roc_auc', 'r2']
        return any(metric in self.scoring for metric in higher_is_better)
    
    def _decode_parameters(self, x: np.ndarray) -> Dict[str, Any]:
        """Decode optimization vector to SVM parameters."""
        params = {}
        
        for i, param_name in enumerate(self.optimize_params):
            value = x[i]
            
            if param_name in ['C', 'gamma']:
                # Use log scale for C and gamma
                if param_name == 'C':
                    params[param_name] = 10 ** np.interp(value, [0, 1], [np.log10(0.001), np.log10(1000)])
                else:  # gamma
                    params[param_name] = 10 ** np.interp(value, [0, 1], [np.log10(1e-6), np.log10(10)])
            elif param_name == 'degree':
                # Integer parameter
                params[param_name] = int(np.round(np.interp(value, [0, 1], [1, 10])))
            elif param_name == 'epsilon':
                params[param_name] = np.interp(value, [0, 1], [0.001, 1.0])
            else:
                # Linear scaling for other parameters
                bounds = self.bounds[i]
                params[param_name] = np.interp(value, [0, 1], bounds)
        
        return params
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate SVM with given hyperparameters.
        
        Args:
            x: Parameter vector (normalized to [0, 1])
        
        Returns:
            Negative cross-validation score (for minimization)
        """
        self.evaluation_count += 1
        
        try:
            # Decode parameters
            params = self._decode_parameters(x)
            
            # Set default parameters
            if self.problem_type == 'classification':
                svm_params = {
                    'kernel': self.kernel_type,
                    'random_state': self.random_state,
                    'max_iter': 1000
                }
            else:
                svm_params = {
                    'kernel': self.kernel_type,
                    'max_iter': 1000
                }
            
            # Update with optimized parameters
            svm_params.update(params)
            
            # Create SVM model
            if self.problem_type == 'classification':
                model = SVC(**svm_params)
            else:
                model = SVR(**svm_params)
            
            # Perform cross-validation
            scores = cross_val_score(
                model, self.X_train_scaled, self.y_train,
                cv=self.cv, scoring=self.scoring, n_jobs=-1
            )
            
            mean_score = np.mean(scores)
            
            # Update best parameters
            is_better = (mean_score > self.best_score if self._is_score_better_when_higher() 
                        else mean_score < self.best_score)
            
            if is_better:
                self.best_score = mean_score
                self.best_params = params.copy()
            
            # Return negative score for minimization (ABC minimizes by default)
            return -mean_score if self._is_score_better_when_higher() else mean_score
            
        except Exception as e:
            # Return a large penalty for invalid parameters
            return 1000.0
    
    def get_best_model(self) -> Union[SVC, SVR]:
        """Get the best trained model."""
        if self.best_params is None:
            raise ValueError("No optimization has been performed yet")
        
        # Create and train the best model
        if self.problem_type == 'classification':
            model = SVC(
                kernel=self.kernel_type,
                random_state=self.random_state,
                max_iter=1000,
                **self.best_params
            )
        else:
            model = SVR(
                kernel=self.kernel_type,
                max_iter=1000,
                **self.best_params
            )
        
        model.fit(self.X_train_scaled, self.y_train)
        return model
    
    def evaluate_best_model(self) -> Dict[str, float]:
        """Evaluate the best model on test set."""
        model = self.get_best_model()
        y_pred = model.predict(self.X_test_scaled)
        
        results = {
            'best_params': self.best_params,
            'cv_score': self.best_score,
            'evaluation_count': self.evaluation_count
        }
        
        if self.problem_type == 'classification':
            results.update({
                'test_accuracy': accuracy_score(self.y_test, y_pred),
                'test_f1': f1_score(self.y_test, y_pred, average='weighted'),
                'test_precision': precision_score(self.y_test, y_pred, average='weighted'),
                'test_recall': recall_score(self.y_test, y_pred, average='weighted')
            })
        else:
            results.update({
                'test_mse': mean_squared_error(self.y_test, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred))
            })
        
        return results


class SVMDatasetLoader:
    """Utility class for loading common datasets for SVM optimization experiments."""
    
    @staticmethod
    def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Load a dataset by name.
        
        Args:
            name: Dataset name
        
        Returns:
            X, y, problem_type
        """
        datasets = {
            'iris': (load_iris, 'classification'),
            'wine': (load_wine, 'classification'),
            'breast_cancer': (load_breast_cancer, 'classification'),
            'diabetes': (load_diabetes, 'regression'),
        }
        
        # California housing dataset as replacement for Boston
        try:
            from sklearn.datasets import fetch_california_housing
            datasets['california_housing'] = (fetch_california_housing, 'regression')
        except ImportError:
            pass
        
        if name.lower() not in datasets:
            available = ', '.join(datasets.keys())
            raise ValueError(f"Dataset '{name}' not available. Available datasets: {available}")
        
        loader, problem_type = datasets[name.lower()]
        data = loader()
        
        return data.data, data.target, problem_type
    
    @staticmethod
    def get_available_datasets() -> Dict[str, str]:
        """Get list of available datasets."""
        datasets = {
            'iris': 'classification',
            'wine': 'classification', 
            'breast_cancer': 'classification',
            'diabetes': 'regression'
        }
        
        # Check if California housing is available
        try:
            from sklearn.datasets import fetch_california_housing
            datasets['california_housing'] = 'regression'
        except ImportError:
            pass
        
        return datasets


def create_svm_problem(dataset_name: str, 
                      kernel: str = 'rbf',
                      optimize_params: List[str] = None,
                      **kwargs) -> SVMOptimizationProblem:
    """
    Convenience function to create SVM optimization problem.
    
    Args:
        dataset_name: Name of the dataset to load
        kernel: SVM kernel type
        optimize_params: Parameters to optimize
        **kwargs: Additional arguments for SVMOptimizationProblem
    
    Returns:
        SVMOptimizationProblem instance
    """
    X, y, problem_type = SVMDatasetLoader.load_dataset(dataset_name)
    
    return SVMOptimizationProblem(
        X=X, y=y, 
        problem_type=problem_type,
        kernel_type=kernel,
        optimize_params=optimize_params,
        **kwargs
    )