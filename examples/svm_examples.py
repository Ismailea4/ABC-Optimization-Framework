"""
SVM optimization examples and experiments
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.svm_optimization import SVMOptimizationProblem, SVMDatasetLoader, create_svm_problem
from core.enhanced_abc import EnhancedABC
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error


class SVMExperimentRunner:
    """Class for running SVM optimization experiments."""
    
    def __init__(self):
        self.results = {}
    
    def run_single_experiment(self, 
                            dataset_name: str,
                            algorithm_type: str = 'enhanced',
                            kernel: str = 'rbf',
                            num_bees: int = 30,
                            max_iterations: int = 50,
                            optimize_params: List[str] = None) -> Dict[str, Any]:
        """
        Run a single SVM optimization experiment.
        
        Args:
            dataset_name: Name of the dataset
            algorithm_type: 'basic' or 'enhanced'
            kernel: SVM kernel type
            num_bees: Number of bees in ABC
            max_iterations: Maximum iterations
            optimize_params: Parameters to optimize
        
        Returns:
            Dictionary with experiment results
        """
        print(f"\nRunning SVM optimization on {dataset_name} dataset...")
        print(f"Algorithm: {algorithm_type.upper()} ABC")
        print(f"Kernel: {kernel}")
        print(f"Optimizing parameters: {optimize_params or 'default'}")
        
        # Create SVM problem
        svm_problem = create_svm_problem(
            dataset_name=dataset_name,
            kernel=kernel,
            optimize_params=optimize_params
        )
        
        # Get baseline performance (default SVM)
        baseline_score = self._get_baseline_performance(svm_problem)
        
        # Create ABC algorithm
        if algorithm_type.lower() == 'enhanced':
            abc = EnhancedABC(
                num_bees=num_bees,
                bounds=svm_problem.bounds,
                objective_function=svm_problem.evaluate,
                max_iterations=max_iterations,
                use_sa=True,
                use_pbsa=True,
                pbsa_interval=10
            )
        else:  # basic ABC
            # Note: You'll need to implement a basic ABC class or modify EnhancedABC
            abc = EnhancedABC(
                num_bees=num_bees,
                bounds=svm_problem.bounds,
                objective_function=svm_problem.evaluate,
                max_iterations=max_iterations,
                use_sa=False,
                use_pbsa=False
            )
        
        # Run optimization
        start_time = time.time()
        best_solution, best_fitness = abc.optimize()
        optimization_time = time.time() - start_time
        
        # Evaluate best model
        best_results = svm_problem.evaluate_best_model()
        
        # Compile results
        results = {
            'dataset': dataset_name,
            'algorithm': algorithm_type,
            'kernel': kernel,
            'optimized_params': optimize_params or f"default_{kernel}",
            'baseline_score': baseline_score,
            'optimized_cv_score': best_results['cv_score'],
            'optimization_time': optimization_time,
            'function_evaluations': svm_problem.evaluation_count,
            'best_parameters': best_results['best_params'],
            'test_results': {k: v for k, v in best_results.items() 
                           if k.startswith('test_')},
            'convergence_history': abc.convergence_history
        }
        
        # Calculate improvement
        if svm_problem._is_score_better_when_higher():
            improvement = best_results['cv_score'] - baseline_score
            improvement_pct = (improvement / abs(baseline_score)) * 100
        else:
            improvement = baseline_score - best_results['cv_score']
            improvement_pct = (improvement / abs(baseline_score)) * 100
        
        results['improvement'] = improvement
        results['improvement_percentage'] = improvement_pct
        
        self._print_results(results)
        
        return results
    
    def _get_baseline_performance(self, svm_problem: SVMOptimizationProblem) -> float:
        """Get baseline SVM performance with default parameters."""
        if svm_problem.problem_type == 'classification':
            model = SVC(kernel=svm_problem.kernel_type, random_state=svm_problem.random_state)
        else:
            model = SVR(kernel=svm_problem.kernel_type)
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            model, svm_problem.X_train_scaled, svm_problem.y_train,
            cv=svm_problem.cv, scoring=svm_problem.scoring, n_jobs=-1
        )
        
        return np.mean(scores)
    
    def _print_results(self, results: Dict[str, Any]):
        """Print experiment results."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Dataset: {results['dataset']}")
        print(f"Algorithm: {results['algorithm'].upper()} ABC")
        print(f"Kernel: {results['kernel']}")
        print(f"Optimized Parameters: {results['optimized_params']}")
        print(f"\nPerformance:")
        print(f"  Baseline CV Score: {results['baseline_score']:.4f}")
        print(f"  Optimized CV Score: {results['optimized_cv_score']:.4f}")
        print(f"  Improvement: {results['improvement']:.4f} ({results['improvement_percentage']:.2f}%)")
        print(f"\nOptimization Details:")
        print(f"  Time: {results['optimization_time']:.2f} seconds")
        print(f"  Function Evaluations: {results['function_evaluations']}")
        print(f"  Best Parameters: {results['best_parameters']}")
        print(f"\nTest Set Results:")
        for metric, value in results['test_results'].items():
            print(f"  {metric}: {value:.4f}")
        print(f"{'='*60}\n")
    
    def compare_algorithms(self, 
                          dataset_name: str,
                          kernels: List[str] = ['rbf', 'linear'],
                          algorithms: List[str] = ['basic', 'enhanced'],
                          num_runs: int = 3) -> pd.DataFrame:
        """
        Compare different algorithms and kernels.
        
        Args:
            dataset_name: Name of the dataset
            kernels: List of kernels to test
            algorithms: List of algorithms to test
            num_runs: Number of runs for each configuration
        
        Returns:
            DataFrame with comparison results
        """
        print(f"\nRunning algorithm comparison on {dataset_name} dataset...")
        
        results = []
        
        for kernel in kernels:
            for algorithm in algorithms:
                print(f"\nTesting {algorithm.upper()} ABC with {kernel} kernel...")
                
                run_results = []
                for run in range(num_runs):
                    print(f"  Run {run + 1}/{num_runs}...")
                    
                    result = self.run_single_experiment(
                        dataset_name=dataset_name,
                        algorithm_type=algorithm,
                        kernel=kernel,
                        num_bees=20,  # Smaller for faster comparison
                        max_iterations=30
                    )
                    
                    run_results.append(result)
                
                # Aggregate results
                avg_improvement = np.mean([r['improvement'] for r in run_results])
                std_improvement = np.std([r['improvement'] for r in run_results])
                avg_time = np.mean([r['optimization_time'] for r in run_results])
                avg_evaluations = np.mean([r['function_evaluations'] for r in run_results])
                
                results.append({
                    'dataset': dataset_name,
                    'algorithm': algorithm,
                    'kernel': kernel,
                    'avg_improvement': avg_improvement,
                    'std_improvement': std_improvement,
                    'avg_time': avg_time,
                    'avg_evaluations': avg_evaluations,
                    'runs': num_runs
                })
        
        df = pd.DataFrame(results)
        self._print_comparison_results(df)
        
        return df
    
    def _print_comparison_results(self, df: pd.DataFrame):
        """Print comparison results table."""
        print(f"\n{'='*80}")
        print(f"ALGORITHM COMPARISON RESULTS")
        print(f"{'='*80}")
        print(df.to_string(index=False, float_format='%.4f'))
        print(f"{'='*80}\n")
    
    def run_dataset_comparison(self, 
                             datasets: List[str] = None,
                             algorithm: str = 'enhanced') -> pd.DataFrame:
        """
        Compare performance across different datasets.
        
        Args:
            datasets: List of dataset names
            algorithm: Algorithm type to use
        
        Returns:
            DataFrame with dataset comparison results
        """
        if datasets is None:
            datasets = list(SVMDatasetLoader.get_available_datasets().keys())
        
        print(f"\nRunning dataset comparison with {algorithm.upper()} ABC...")
        
        results = []
        
        for dataset in datasets:
            try:
                print(f"\nTesting on {dataset} dataset...")
                
                result = self.run_single_experiment(
                    dataset_name=dataset,
                    algorithm_type=algorithm,
                    kernel='rbf',  # Use RBF for all datasets
                    num_bees=25,
                    max_iterations=40
                )
                
                results.append({
                    'dataset': dataset,
                    'problem_type': SVMDatasetLoader.get_available_datasets()[dataset],
                    'improvement': result['improvement'],
                    'improvement_pct': result['improvement_percentage'],
                    'optimization_time': result['optimization_time'],
                    'function_evaluations': result['function_evaluations']
                })
                
            except Exception as e:
                print(f"Error with dataset {dataset}: {e}")
                continue
        
        df = pd.DataFrame(results)
        self._print_dataset_comparison(df)
        
        return df
    
    def _print_dataset_comparison(self, df: pd.DataFrame):
        """Print dataset comparison results."""
        print(f"\n{'='*80}")
        print(f"DATASET COMPARISON RESULTS")
        print(f"{'='*80}")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"Average Improvement: {df['improvement'].mean():.4f}")
        print(f"Average Improvement %: {df['improvement_pct'].mean():.2f}%")
        print(f"Average Time: {df['optimization_time'].mean():.2f} seconds")
        print(f"{'='*80}\n")


def plot_convergence_comparison(results: List[Dict[str, Any]], save_path: str = None):
    """Plot convergence comparison for multiple experiments."""
    plt.figure(figsize=(12, 8))
    
    for result in results:
        label = f"{result['algorithm'].upper()} ABC ({result['kernel']})"
        convergence = result['convergence_history']
        
        if convergence:
            plt.plot(convergence, label=label, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Comparison - SVM Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Example usage functions
def quick_example():
    """Quick example of SVM optimization."""
    runner = SVMExperimentRunner()
    
    # Run a simple experiment
    result = runner.run_single_experiment(
        dataset_name='iris',
        algorithm_type='enhanced',
        kernel='rbf',
        num_bees=20,
        max_iterations=30
    )
    
    return result


def full_comparison_example():
    """Full comparison across algorithms and datasets."""
    runner = SVMExperimentRunner()
    
    # Compare algorithms on iris dataset
    algo_comparison = runner.compare_algorithms(
        dataset_name='iris',
        kernels=['rbf', 'linear'],
        algorithms=['basic', 'enhanced'],
        num_runs=2
    )
    
    # Compare across datasets
    dataset_comparison = runner.run_dataset_comparison(
        datasets=['iris', 'wine', 'breast_cancer'],
        algorithm='enhanced'
    )
    
    return algo_comparison, dataset_comparison


if __name__ == "__main__":
    print("SVM Optimization with ABC Algorithms")
    print("=" * 50)
    
    # Run examples
    print("\n1. Quick Example:")
    quick_example()
    
    print("\n2. Full Comparison:")
    full_comparison_example()