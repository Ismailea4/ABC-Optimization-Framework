"""
Experiment runner for comparing ABC algorithms with SVM optimization
"""
import time
import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.enhanced_abc import EnhancedABC
from core.rl_abc import RLABC
from problems.svm_optimization import create_svm_problem, SVMDatasetLoader


class ExperimentRunner:
    """
    Class for running comparative experiments between ABC variants on SVM optimization.
    """
    
    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize experiment runner.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def compare_abc_algorithms(self, 
                              dataset_name: str = 'iris',
                              kernel: str = 'rbf',
                              num_bees: int = 20,
                              max_iterations: int = 30,
                              num_runs: int = 3) -> Dict[str, Any]:
        """
        Compare Basic ABC, Enhanced ABC, and RL-ABC on SVM optimization.
        
        Args:
            dataset_name: Dataset to use for comparison
            kernel: SVM kernel type
            num_bees: Number of bees in population
            max_iterations: Maximum iterations per run
            num_runs: Number of independent runs
        
        Returns:
            Dictionary with comparison results
        """
        print(f"Comparing ABC algorithms on {dataset_name} dataset...")
        print(f"Configuration: {num_bees} bees, {max_iterations} iterations, {num_runs} runs")
        
        algorithms = {
            'Basic ABC': {
                'class': EnhancedABC,
                'params': {
                    'use_sa': False,
                    'use_pbsa': False
                }
            },
            'Enhanced ABC': {
                'class': EnhancedABC,
                'params': {
                    'use_sa': True,
                    'use_pbsa': True,
                    'pbsa_interval': 10
                }
            },
            'RL-ABC': {
                'class': RLABC,
                'params': {
                    'use_sa': True,
                    'use_pbsa': False,  # Let RL handle the strategy selection
                    'rl_learning_rate': 0.1,
                    'rl_epsilon': 0.3
                }
            }
        }
        
        results = {}
        
        for alg_name, alg_config in algorithms.items():
            print(f"\nTesting {alg_name}...")
            
            alg_results = {
                'accuracies': [],
                'improvements': [],
                'times': [],
                'convergence_histories': [],
                'best_params': [],
                'evaluations': []
            }
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...")
                
                # Create fresh SVM problem for each run
                svm_problem = create_svm_problem(
                    dataset_name=dataset_name,
                    kernel=kernel,
                    optimize_params=['C', 'gamma']
                )
                
                # Get baseline performance
                baseline_score = self._get_baseline_svm_performance(svm_problem)
                
                # Create algorithm instance
                abc_algorithm = alg_config['class'](
                    num_bees=num_bees,
                    bounds=svm_problem.bounds,
                    objective_function=svm_problem.evaluate,
                    max_iterations=max_iterations,
                    **alg_config['params']
                )
                
                # Run optimization
                start_time = time.time()
                best_solution, best_fitness = abc_algorithm.optimize()
                optimization_time = time.time() - start_time
                
                # Get results
                model_results = svm_problem.evaluate_best_model()
                
                # Store results
                alg_results['accuracies'].append(model_results['test_accuracy'])
                improvement = ((model_results['cv_score'] - baseline_score) / abs(baseline_score)) * 100
                alg_results['improvements'].append(improvement)
                alg_results['times'].append(optimization_time)
                alg_results['convergence_histories'].append(abc_algorithm.convergence_history.copy())
                alg_results['best_params'].append(model_results['best_params'])
                alg_results['evaluations'].append(model_results['evaluation_count'])
            
            # Calculate statistics
            alg_results['mean_accuracy'] = np.mean(alg_results['accuracies'])
            alg_results['std_accuracy'] = np.std(alg_results['accuracies'])
            alg_results['mean_improvement'] = np.mean(alg_results['improvements'])
            alg_results['std_improvement'] = np.std(alg_results['improvements'])
            alg_results['mean_time'] = np.mean(alg_results['times'])
            alg_results['mean_evaluations'] = np.mean(alg_results['evaluations'])
            
            results[alg_name] = alg_results
            
            print(f"  Results: Accuracy={alg_results['mean_accuracy']:.4f}±{alg_results['std_accuracy']:.4f}, "
                  f"Improvement={alg_results['mean_improvement']:.2f}±{alg_results['std_improvement']:.2f}%")
        
        # Add experiment metadata
        results['metadata'] = {
            'dataset': dataset_name,
            'kernel': kernel,
            'num_bees': num_bees,
            'max_iterations': max_iterations,
            'num_runs': num_runs,
            'baseline_score': baseline_score
        }
        
        return results
    
    def _get_baseline_svm_performance(self, svm_problem) -> float:
        """Get baseline SVM performance with default parameters."""
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        
        if svm_problem.problem_type == 'classification':
            model = SVC(kernel=svm_problem.kernel_type, random_state=svm_problem.random_state)
        else:
            from sklearn.svm import SVR
            model = SVR(kernel=svm_problem.kernel_type)
        
        scores = cross_val_score(
            model, svm_problem.X_train_scaled, svm_problem.y_train,
            cv=svm_problem.cv, scoring=svm_problem.scoring, n_jobs=-1
        )
        
        return np.mean(scores)
    
    def compare_across_datasets(self, 
                               datasets: List[str] = None,
                               algorithm: str = 'Enhanced ABC',
                               num_runs: int = 2) -> pd.DataFrame:
        """
        Compare algorithm performance across multiple datasets.
        
        Args:
            datasets: List of dataset names to test
            algorithm: Algorithm to use for comparison
            num_runs: Number of runs per dataset
        
        Returns:
            DataFrame with results across datasets
        """
        if datasets is None:
            datasets = ['iris', 'wine', 'breast_cancer']
        
        print(f"Comparing {algorithm} across datasets: {datasets}")
        
        results = []
        
        for dataset in datasets:
            try:
                print(f"\nTesting on {dataset}...")
                
                comparison_results = self.compare_abc_algorithms(
                    dataset_name=dataset,
                    num_runs=num_runs,
                    num_bees=15,  # Smaller for faster comparison
                    max_iterations=20
                )
                
                if algorithm in comparison_results:
                    alg_result = comparison_results[algorithm]
                    
                    results.append({
                        'dataset': dataset,
                        'mean_accuracy': alg_result['mean_accuracy'],
                        'std_accuracy': alg_result['std_accuracy'],
                        'mean_improvement': alg_result['mean_improvement'],
                        'std_improvement': alg_result['std_improvement'],
                        'mean_time': alg_result['mean_time'],
                        'problem_type': SVMDatasetLoader.get_available_datasets()[dataset]
                    })
                
            except Exception as e:
                print(f"Error with dataset {dataset}: {e}")
                continue
        
        df = pd.DataFrame(results)
        return df
    
    def run_rl_analysis(self, 
                       dataset_name: str = 'iris',
                       num_bees: int = 20,
                       max_iterations: int = 50) -> Dict[str, Any]:
        """
        Run detailed analysis of RL-ABC algorithm.
        
        Args:
            dataset_name: Dataset to use
            num_bees: Number of bees
            max_iterations: Number of iterations
        
        Returns:
            Dictionary with RL analysis results
        """
        print(f"Running RL-ABC analysis on {dataset_name}...")
        
        svm_problem = create_svm_problem(
            dataset_name=dataset_name,
            kernel='rbf',
            optimize_params=['C', 'gamma']
        )
        
        rl_abc = RLABC(
            num_bees=num_bees,
            bounds=svm_problem.bounds,
            objective_function=svm_problem.evaluate,
            max_iterations=max_iterations,
            use_sa=True,
            use_pbsa=False,
            rl_learning_rate=0.1,
            rl_epsilon=0.5  # Higher exploration for analysis
        )
        
        # Run optimization
        start_time = time.time()
        best_solution, best_fitness = rl_abc.optimize()
        optimization_time = time.time() - start_time
        
        # Get SVM results
        model_results = svm_problem.evaluate_best_model()
        
        # Get RL statistics
        rl_stats = rl_abc.get_rl_statistics()
        
        return {
            'svm_results': model_results,
            'rl_statistics': rl_stats,
            'convergence_history': rl_abc.convergence_history,
            'optimization_time': optimization_time,
            'best_fitness': best_fitness
        }