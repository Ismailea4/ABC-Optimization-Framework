"""
Simple SVM optimization example
Demonstrates how to use the ABC framework for SVM hyperparameter optimization
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from problems.svm_optimization import create_svm_problem
from core.enhanced_abc import EnhancedABC
import time

def main():
    print("🐝 ABC-SVM Optimization Example")
    print("=" * 50)
    
    # Example 1: Quick optimization on iris dataset
    print("\n📊 Example 1: Iris Dataset Optimization")
    print("-" * 30)
    
    # Create SVM problem
    svm_problem = create_svm_problem(
        dataset_name='iris',
        kernel='rbf',
        optimize_params=['C', 'gamma']
    )
    
    print(f"Dataset: Iris (classification)")
    print(f"Features: {svm_problem.X.shape[1]}, Samples: {svm_problem.X.shape[0]}")
    print(f"Optimizing: C and gamma parameters")
    
    # Get baseline performance
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    baseline_model = SVC(kernel='rbf', random_state=42)
    baseline_scores = cross_val_score(
        baseline_model, svm_problem.X_train_scaled, svm_problem.y_train,
        cv=svm_problem.cv, scoring='accuracy'
    )
    baseline_score = baseline_scores.mean()
    
    print(f"Baseline accuracy: {baseline_score:.4f}")
    
    # Run ABC optimization
    print(f"\nRunning ABC optimization...")
    start_time = time.time()
    
    abc = EnhancedABC(
        num_bees=25,
        bounds=svm_problem.bounds,
        objective_function=svm_problem.evaluate,
        max_iterations=40,
        use_sa=True,
        use_pbsa=True,
        pbsa_interval=10
    )
    
    best_solution, best_fitness = abc.optimize()
    optimization_time = time.time() - start_time
    
    # Get results
    results = svm_problem.evaluate_best_model()
    
    print(f"\n✅ Optimization Results:")
    print(f"Best parameters: C={results['best_params']['C']:.3f}, gamma={results['best_params']['gamma']:.6f}")
    print(f"Optimized CV accuracy: {results['cv_score']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Improvement: {((results['cv_score'] - baseline_score) / baseline_score * 100):+.2f}%")
    print(f"Optimization time: {optimization_time:.2f} seconds")
    print(f"Function evaluations: {results['evaluation_count']}")
    
    # Example 2: Different dataset
    print(f"\n📊 Example 2: Wine Dataset Optimization")
    print("-" * 30)
    
    wine_problem = create_svm_problem(
        dataset_name='wine',
        kernel='rbf',
        optimize_params=['C', 'gamma']
    )
    
    print(f"Dataset: Wine (classification)")
    print(f"Features: {wine_problem.X.shape[1]}, Samples: {wine_problem.X.shape[0]}")
    
    # Quick optimization
    abc_wine = EnhancedABC(
        num_bees=20,
        bounds=wine_problem.bounds,
        objective_function=wine_problem.evaluate,
        max_iterations=30,
        use_sa=True,
        use_pbsa=False  # Disable PBSA for quicker run
    )
    
    print(f"Running optimization (20 bees, 30 iterations)...")
    start_time = time.time()
    best_solution, best_fitness = abc_wine.optimize()
    wine_time = time.time() - start_time
    
    wine_results = wine_problem.evaluate_best_model()
    
    print(f"\n✅ Wine Dataset Results:")
    print(f"Best parameters: C={wine_results['best_params']['C']:.3f}, gamma={wine_results['best_params']['gamma']:.6f}")
    print(f"Test accuracy: {wine_results['test_accuracy']:.4f}")
    print(f"Optimization time: {wine_time:.2f} seconds")
    
    # Example 3: Regression with diabetes dataset
    print(f"\n📊 Example 3: Diabetes Dataset (Regression)")
    print("-" * 30)
    
    diabetes_problem = create_svm_problem(
        dataset_name='diabetes',
        kernel='rbf',
        optimize_params=['C', 'gamma', 'epsilon']  # Include epsilon for SVR
    )
    
    print(f"Dataset: Diabetes (regression)")
    print(f"Features: {diabetes_problem.X.shape[1]}, Samples: {diabetes_problem.X.shape[0]}")
    print(f"Optimizing: C, gamma, and epsilon parameters")
    
    abc_diabetes = EnhancedABC(
        num_bees=20,
        bounds=diabetes_problem.bounds,
        objective_function=diabetes_problem.evaluate,
        max_iterations=25,
        use_sa=True,
        use_pbsa=False
    )
    
    print(f"Running optimization...")
    start_time = time.time()
    best_solution, best_fitness = abc_diabetes.optimize()
    diabetes_time = time.time() - start_time
    
    diabetes_results = diabetes_problem.evaluate_best_model()
    
    print(f"\n✅ Diabetes Dataset Results:")
    print(f"Best parameters: {diabetes_results['best_params']}")
    print(f"Test RMSE: {diabetes_results['test_rmse']:.4f}")
    print(f"Optimization time: {diabetes_time:.2f} seconds")
    
    print(f"\n🎉 All examples completed successfully!")
    print(f"Total time: {optimization_time + wine_time + diabetes_time:.2f} seconds")
    
    # Summary
    print(f"\n📈 Summary")
    print("=" * 50)
    print(f"{'Dataset':<15} {'Type':<13} {'Test Score':<12} {'Time (s)':<10}")
    print("-" * 50)
    print(f"{'Iris':<15} {'Classification':<13} {results['test_accuracy']:<12.4f} {optimization_time:<10.2f}")
    print(f"{'Wine':<15} {'Classification':<13} {wine_results['test_accuracy']:<12.4f} {wine_time:<10.2f}")
    print(f"{'Diabetes':<15} {'Regression':<13} {diabetes_results['test_rmse']:<12.4f} {diabetes_time:<10.2f}")
    print("=" * 50)
    
    print(f"\n💡 Tips for better results:")
    print(f"   • Increase num_bees (30-50) and max_iterations (50-100) for better optimization")
    print(f"   • Enable both SA and PBSA for enhanced performance")
    print(f"   • Try different kernels (linear, poly, sigmoid) for different datasets")
    print(f"   • Use cross-validation for robust performance estimates")


if __name__ == "__main__":
    main()