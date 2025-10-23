"""
Comprehensive experiment script for ABC algorithms with SVM optimization
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from experiments.run_experiment import ExperimentRunner
from experiments.visualize_results import ResultsVisualizer
import matplotlib.pyplot as plt

def main():
    """Run comprehensive ABC-SVM experiments."""
    print("🔬 ABC-SVM Comprehensive Experiments")
    print("=" * 60)
    
    # Initialize components
    runner = ExperimentRunner(random_seed=42)
    visualizer = ResultsVisualizer()
    
    # Experiment 1: Algorithm Comparison
    print("\n📊 Experiment 1: Algorithm Comparison on Iris Dataset")
    print("-" * 50)
    
    results = runner.compare_abc_algorithms(
        dataset_name='iris',
        kernel='rbf',
        num_bees=15,
        max_iterations=20,
        num_runs=2  # Small for quick demo
    )
    
    print("\nVisualizing comparison results...")
    visualizer.plot_abc_comparison(results, save_path='abc_comparison.png')
    visualizer.plot_convergence_comparison(results, save_path='convergence_comparison.png')
    
    # Experiment 2: Dataset Comparison
    print("\n📊 Experiment 2: Enhanced ABC Across Datasets")
    print("-" * 50)
    
    dataset_df = runner.compare_across_datasets(
        datasets=['iris', 'wine', 'breast_cancer'],
        algorithm='Enhanced ABC',
        num_runs=2
    )
    
    print("\nDataset comparison results:")
    print(dataset_df.to_string(index=False, float_format='%.4f'))
    
    print("\nVisualizing dataset comparison...")
    visualizer.plot_dataset_comparison(dataset_df, save_path='dataset_comparison.png')
    
    # Experiment 3: RL-ABC Analysis
    print("\n🤖 Experiment 3: RL-ABC Detailed Analysis")
    print("-" * 50)
    
    rl_results = runner.run_rl_analysis(
        dataset_name='wine',
        num_bees=15,
        max_iterations=25
    )
    
    print(f"RL-ABC Results:")
    print(f"  SVM Test Accuracy: {rl_results['svm_results']['test_accuracy']:.4f}")
    print(f"  Best Parameters: {rl_results['svm_results']['best_params']}")
    print(f"  RL Epsilon (final): {rl_results['rl_statistics']['epsilon']:.3f}")
    print(f"  Action Preferences: {rl_results['rl_statistics']['action_preferences']}")
    
    print("\nVisualizing RL-ABC analysis...")
    visualizer.plot_rl_analysis(rl_results, save_path='rl_abc_analysis.png')
    
    # Summary
    print("\n📈 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    # Best performing algorithm from experiment 1
    best_alg = max(results.keys() - {'metadata'}, 
                  key=lambda alg: results[alg]['mean_accuracy'])
    print(f"Best performing algorithm: {best_alg}")
    print(f"  Accuracy: {results[best_alg]['mean_accuracy']:.4f} ± {results[best_alg]['std_accuracy']:.4f}")
    print(f"  Improvement: {results[best_alg]['mean_improvement']:.2f}% ± {results[best_alg]['std_improvement']:.2f}%")
    
    # Best dataset performance
    best_dataset_idx = dataset_df['mean_accuracy'].idxmax()
    best_dataset = dataset_df.loc[best_dataset_idx]
    print(f"\nBest dataset performance: {best_dataset['dataset']}")
    print(f"  Accuracy: {best_dataset['mean_accuracy']:.4f} ± {best_dataset['std_accuracy']:.4f}")
    print(f"  Improvement: {best_dataset['mean_improvement']:.2f}%")
    
    # RL insights
    rl_prefs = rl_results['rl_statistics']['action_preferences']
    action_names = ['Standard Search', 'Enhanced Exploration', 'Enhanced Exploitation']
    preferred_action = action_names[rl_prefs.index(max(rl_prefs))]
    print(f"\nRL-ABC preferred strategy: {preferred_action}")
    print(f"  Q-value: {max(rl_prefs):.3f}")
    
    print("\n✅ All experiments completed successfully!")
    print("Generated plots:")
    print("  - abc_comparison.png")
    print("  - convergence_comparison.png") 
    print("  - dataset_comparison.png")
    print("  - rl_abc_analysis.png")


def quick_demo():
    """Quick demonstration of RL-ABC vs Enhanced ABC."""
    print("🚀 Quick Demo: RL-ABC vs Enhanced ABC")
    print("=" * 50)
    
    runner = ExperimentRunner(random_seed=42)
    
    # Compare just RL-ABC vs Enhanced ABC
    results = runner.compare_abc_algorithms(
        dataset_name='iris',
        kernel='rbf',
        num_bees=10,
        max_iterations=15,
        num_runs=1
    )
    
    # Filter to just these two algorithms
    filtered_results = {
        'Enhanced ABC': results['Enhanced ABC'],
        'RL-ABC': results['RL-ABC'],
        'metadata': results['metadata']
    }
    
    print("\nQuick Comparison Results:")
    for alg_name in ['Enhanced ABC', 'RL-ABC']:
        alg_result = results[alg_name]
        print(f"{alg_name}:")
        print(f"  Accuracy: {alg_result['mean_accuracy']:.4f}")
        print(f"  Improvement: {alg_result['mean_improvement']:.2f}%")
        print(f"  Time: {alg_result['mean_time']:.1f}s")
    
    # Simple visualization
    visualizer = ResultsVisualizer()
    visualizer.plot_abc_comparison(filtered_results)
    
    print("\n✅ Quick demo completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ABC-SVM experiments')
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                       help='Experiment mode: full or quick demo')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        main()
    else:
        quick_demo()