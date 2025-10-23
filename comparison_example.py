"""
Comparison example: Basic ABC vs Enhanced ABC for SVM optimization
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from problems.svm_optimization import create_svm_problem
from core.enhanced_abc import EnhancedABC
import time
import numpy as np

def run_comparison():
    print("🔬 ABC Algorithm Comparison for SVM Optimization")
    print("=" * 60)
    
    # Use a smaller dataset for quicker comparison
    dataset_name = 'iris'
    svm_problem = create_svm_problem(
        dataset_name=dataset_name,
        kernel='rbf',
        optimize_params=['C', 'gamma']
    )
    
    print(f"Dataset: {dataset_name.title()}")
    print(f"Problem: SVM hyperparameter optimization (C, gamma)")
    print(f"Configuration: 15 bees, 20 iterations (quick comparison)")
    print()
    
    results = {}
    
    # Test Basic ABC (Enhanced ABC with features disabled)
    print("🐝 Testing Basic ABC...")
    basic_abc = EnhancedABC(
        num_bees=15,
        bounds=svm_problem.bounds,
        objective_function=svm_problem.evaluate,
        max_iterations=20,
        use_sa=False,   # Disable Simulated Annealing
        use_pbsa=False  # Disable Population-Based SA
    )
    
    start_time = time.time()
    best_solution_basic, best_fitness_basic = basic_abc.optimize()
    basic_time = time.time() - start_time
    
    basic_results = svm_problem.evaluate_best_model()
    
    results['basic'] = {
        'time': basic_time,
        'fitness': best_fitness_basic,
        'params': basic_results['best_params'],
        'cv_score': basic_results['cv_score'],
        'test_accuracy': basic_results['test_accuracy'],
        'evaluations': basic_results['evaluation_count'],
        'convergence': basic_abc.convergence_history.copy()
    }
    
    print(f"   ✓ Completed in {basic_time:.2f}s")
    print(f"   ✓ Best fitness: {best_fitness_basic:.4f}")
    print(f"   ✓ Test accuracy: {basic_results['test_accuracy']:.4f}")
    print(f"   ✓ Function evaluations: {basic_results['evaluation_count']}")
    print()
    
    # Test Enhanced ABC 
    print("🚀 Testing Enhanced ABC (with SA + PBSA)...")
    
    # Create new problem instance for enhanced ABC
    svm_problem_enhanced = create_svm_problem(
        dataset_name=dataset_name,
        kernel='rbf',
        optimize_params=['C', 'gamma']
    )
    
    enhanced_abc = EnhancedABC(
        num_bees=15,
        bounds=svm_problem_enhanced.bounds,
        objective_function=svm_problem_enhanced.evaluate,
        max_iterations=20,
        use_sa=True,     # Enable Simulated Annealing
        use_pbsa=True,   # Enable Population-Based SA
        pbsa_interval=5  # Apply PBSA every 5 iterations
    )
    
    start_time = time.time()
    best_solution_enhanced, best_fitness_enhanced = enhanced_abc.optimize()
    enhanced_time = time.time() - start_time
    
    enhanced_results = svm_problem_enhanced.evaluate_best_model()
    
    results['enhanced'] = {
        'time': enhanced_time,
        'fitness': best_fitness_enhanced,
        'params': enhanced_results['best_params'],
        'cv_score': enhanced_results['cv_score'],
        'test_accuracy': enhanced_results['test_accuracy'],
        'evaluations': enhanced_results['evaluation_count'],
        'convergence': enhanced_abc.convergence_history.copy()
    }
    
    print(f"   ✓ Completed in {enhanced_time:.2f}s")
    print(f"   ✓ Best fitness: {best_fitness_enhanced:.4f}")
    print(f"   ✓ Test accuracy: {enhanced_results['test_accuracy']:.4f}")
    print(f"   ✓ Function evaluations: {enhanced_results['evaluation_count']}")
    print()
    
    # Comparison Summary
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"{'Metric':<25} {'Basic ABC':<15} {'Enhanced ABC':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Test Accuracy
    acc_improvement = ((results['enhanced']['test_accuracy'] - results['basic']['test_accuracy']) / 
                      results['basic']['test_accuracy']) * 100
    print(f"{'Test Accuracy':<25} {results['basic']['test_accuracy']:<15.4f} {results['enhanced']['test_accuracy']:<15.4f} {acc_improvement:>+13.2f}%")
    
    # CV Score
    cv_improvement = ((results['enhanced']['cv_score'] - results['basic']['cv_score']) / 
                     results['basic']['cv_score']) * 100
    print(f"{'CV Score':<25} {results['basic']['cv_score']:<15.4f} {results['enhanced']['cv_score']:<15.4f} {cv_improvement:>+13.2f}%")
    
    # Best Fitness (lower is better for minimization)
    fitness_improvement = ((results['basic']['fitness'] - results['enhanced']['fitness']) / 
                          abs(results['basic']['fitness'])) * 100
    print(f"{'Best Fitness':<25} {results['basic']['fitness']:<15.4f} {results['enhanced']['fitness']:<15.4f} {fitness_improvement:>+13.2f}%")
    
    # Time
    time_overhead = ((results['enhanced']['time'] - results['basic']['time']) / 
                    results['basic']['time']) * 100
    print(f"{'Optimization Time (s)':<25} {results['basic']['time']:<15.2f} {results['enhanced']['time']:<15.2f} {time_overhead:>+13.2f}%")
    
    # Function Evaluations
    eval_overhead = ((results['enhanced']['evaluations'] - results['basic']['evaluations']) / 
                    results['basic']['evaluations']) * 100
    print(f"{'Function Evaluations':<25} {results['basic']['evaluations']:<15d} {results['enhanced']['evaluations']:<15d} {eval_overhead:>+13.2f}%")
    
    print("=" * 70)
    
    # Best Parameters
    print(f"\n🎯 BEST PARAMETERS FOUND")
    print("-" * 30)
    print(f"Basic ABC:")
    for param, value in results['basic']['params'].items():
        print(f"   {param}: {value:.6f}")
    
    print(f"\nEnhanced ABC:")
    for param, value in results['enhanced']['params'].items():
        print(f"   {param}: {value:.6f}")
    
    # Convergence Analysis
    print(f"\n📈 CONVERGENCE ANALYSIS")
    print("-" * 30)
    
    basic_conv = results['basic']['convergence']
    enhanced_conv = results['enhanced']['convergence']
    
    if basic_conv and enhanced_conv:
        # Final convergence values
        print(f"Basic ABC final fitness: {basic_conv[-1]:.4f}")
        print(f"Enhanced ABC final fitness: {enhanced_conv[-1]:.4f}")
        
        # Convergence speed (iterations to reach 95% of final value)
        def iterations_to_convergence(convergence_history, threshold=0.95):
            if not convergence_history:
                return len(convergence_history)
            final_val = convergence_history[-1]
            target = final_val + (1 - threshold) * abs(final_val)
            for i, val in enumerate(convergence_history):
                if val <= target:  # For minimization problems
                    return i + 1
            return len(convergence_history)
        
        basic_conv_iter = iterations_to_convergence(basic_conv)
        enhanced_conv_iter = iterations_to_convergence(enhanced_conv)
        
        print(f"Basic ABC convergence speed: {basic_conv_iter} iterations")
        print(f"Enhanced ABC convergence speed: {enhanced_conv_iter} iterations")
    
    # Conclusion
    print(f"\n🏆 CONCLUSION")
    print("-" * 30)
    if results['enhanced']['test_accuracy'] > results['basic']['test_accuracy']:
        winner = "Enhanced ABC"
        advantage = "better solution quality"
    elif results['enhanced']['test_accuracy'] == results['basic']['test_accuracy']:
        if results['enhanced']['time'] < results['basic']['time']:
            winner = "Enhanced ABC"
            advantage = "same quality in less time"
        else:
            winner = "Basic ABC"
            advantage = "same quality with less computational overhead"
    else:
        winner = "Basic ABC"
        advantage = "better solution quality"
    
    print(f"Winner: {winner}")
    print(f"Reason: {advantage}")
    
    if winner == "Enhanced ABC":
        print(f"The enhanced features (SA + PBSA) successfully improved the optimization!")
    else:
        print(f"For this particular problem, the basic ABC was sufficient.")
    
    print(f"\n💡 Recommendations:")
    print(f"   • For quick experiments: Use Basic ABC")
    print(f"   • For best results: Use Enhanced ABC with more iterations")
    print(f"   • For production use: Enhanced ABC with 30+ bees and 50+ iterations")
    
    return results

if __name__ == "__main__":
    results = run_comparison()