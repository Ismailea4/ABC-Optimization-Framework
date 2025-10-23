"""
Visualization utilities for ABC experiments with SVM optimization
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional
import seaborn as sns


class ResultsVisualizer:
    """
    Class for visualizing ABC experiment results.
    """
    
    def __init__(self, style: str = 'whitegrid'):
        """
        Initialize visualizer.
        
        Args:
            style: Seaborn style for plots
        """
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11
    
    def plot_abc_comparison(self, 
                           results: Dict[str, Any],
                           save_path: Optional[str] = None) -> None:
        """
        Plot comparison of ABC algorithms on SVM optimization.
        
        Args:
            results: Results from ExperimentRunner.compare_abc_algorithms()
            save_path: Optional path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        algorithms = list(results.keys())
        algorithms = [alg for alg in algorithms if alg != 'metadata']
        
        # Extract data
        accuracies = [results[alg]['mean_accuracy'] for alg in algorithms]
        accuracy_stds = [results[alg]['std_accuracy'] for alg in algorithms]
        improvements = [results[alg]['mean_improvement'] for alg in algorithms]
        improvement_stds = [results[alg]['std_improvement'] for alg in algorithms]
        times = [results[alg]['mean_time'] for alg in algorithms]
        evaluations = [results[alg]['mean_evaluations'] for alg in algorithms]
        
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        # Plot 1: Accuracy comparison
        bars1 = ax1.bar(algorithms, accuracies, yerr=accuracy_stds, 
                       capsize=5, color=colors[:len(algorithms)], alpha=0.8)
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for i, (acc, std) in enumerate(zip(accuracies, accuracy_stds)):
            ax1.text(i, acc + std + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
        
        # Plot 2: Improvement comparison
        bars2 = ax2.bar(algorithms, improvements, yerr=improvement_stds,
                       capsize=5, color=colors[:len(algorithms)], alpha=0.8)
        ax2.set_title('Performance Improvement (%)')
        ax2.set_ylabel('Improvement over Baseline (%)')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        for i, (imp, std) in enumerate(zip(improvements, improvement_stds)):
            ax2.text(i, imp + std + 0.1, f'{imp:.1f}%', ha='center', fontweight='bold')
        
        # Plot 3: Time comparison
        bars3 = ax3.bar(algorithms, times, color=colors[:len(algorithms)], alpha=0.8)
        ax3.set_title('Optimization Time')
        ax3.set_ylabel('Time (seconds)')
        for i, time_val in enumerate(times):
            ax3.text(i, time_val + max(times)*0.02, f'{time_val:.1f}s', ha='center', fontweight='bold')
        
        # Plot 4: Function evaluations
        bars4 = ax4.bar(algorithms, evaluations, color=colors[:len(algorithms)], alpha=0.8)
        ax4.set_title('Function Evaluations')
        ax4.set_ylabel('Number of Evaluations')
        for i, eval_val in enumerate(evaluations):
            ax4.text(i, eval_val + max(evaluations)*0.02, f'{int(eval_val)}', ha='center', fontweight='bold')
        
        # Add dataset info
        if 'metadata' in results:
            metadata = results['metadata']
            fig.suptitle(f"ABC Algorithm Comparison - {metadata['dataset'].title()} Dataset "
                        f"({metadata['kernel']} kernel)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_comparison(self, 
                                  results: Dict[str, Any],
                                  show_std: bool = True,
                                  save_path: Optional[str] = None) -> None:
        """
        Plot convergence curves for ABC algorithm comparison.
        
        Args:
            results: Results from ExperimentRunner.compare_abc_algorithms()
            show_std: Whether to show standard deviation bands
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        algorithms = list(results.keys())
        algorithms = [alg for alg in algorithms if alg != 'metadata']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, alg_name in enumerate(algorithms):
            alg_results = results[alg_name]
            convergence_histories = alg_results['convergence_histories']
            
            if not convergence_histories:
                continue
            
            # Calculate mean and std convergence
            max_length = max(len(hist) for hist in convergence_histories)
            padded_histories = []
            
            for hist in convergence_histories:
                padded = hist + [hist[-1]] * (max_length - len(hist))
                padded_histories.append(padded)
            
            padded_histories = np.array(padded_histories)
            mean_convergence = np.mean(padded_histories, axis=0)
            std_convergence = np.std(padded_histories, axis=0)
            
            iterations = range(len(mean_convergence))
            
            # Plot mean
            plt.plot(iterations, mean_convergence, 
                    color=colors[i % len(colors)], 
                    label=alg_name, linewidth=2)
            
            # Plot std band
            if show_std and len(convergence_histories) > 1:
                plt.fill_between(iterations, 
                               mean_convergence - std_convergence,
                               mean_convergence + std_convergence,
                               color=colors[i % len(colors)], alpha=0.2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (CV Score)')
        plt.title('Convergence Comparison of ABC Algorithms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dataset_comparison(self, 
                               df: pd.DataFrame,
                               save_path: Optional[str] = None) -> None:
        """
        Plot algorithm performance across different datasets.
        
        Args:
            df: DataFrame from ExperimentRunner.compare_across_datasets()
            save_path: Optional path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        datasets = df['dataset'].values
        x_pos = np.arange(len(datasets))
        
        # Plot 1: Mean accuracy
        bars1 = ax1.bar(x_pos, df['mean_accuracy'], yerr=df['std_accuracy'], 
                       capsize=5, color='lightblue', alpha=0.8)
        ax1.set_title('Test Accuracy Across Datasets')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(datasets, rotation=45)
        for i, (acc, std) in enumerate(zip(df['mean_accuracy'], df['std_accuracy'])):
            ax1.text(i, acc + std + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
        
        # Plot 2: Improvement
        colors = ['lightcoral' if imp >= 0 else 'lightgray' for imp in df['mean_improvement']]
        bars2 = ax2.bar(x_pos, df['mean_improvement'], yerr=df['std_improvement'],
                       capsize=5, color=colors, alpha=0.8)
        ax2.set_title('Performance Improvement (%)')
        ax2.set_ylabel('Improvement over Baseline (%)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(datasets, rotation=45)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        for i, (imp, std) in enumerate(zip(df['mean_improvement'], df['std_improvement'])):
            ax2.text(i, imp + std + 0.1, f'{imp:.1f}%', ha='center', fontweight='bold')
        
        # Plot 3: Optimization time
        bars3 = ax3.bar(x_pos, df['mean_time'], color='lightgreen', alpha=0.8)
        ax3.set_title('Optimization Time')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(datasets, rotation=45)
        for i, time_val in enumerate(df['mean_time']):
            ax3.text(i, time_val + df['mean_time'].max()*0.02, f'{time_val:.1f}s', 
                    ha='center', fontweight='bold')
        
        # Plot 4: Problem type distribution
        problem_types = df['problem_type'].value_counts()
        ax4.pie(problem_types.values, labels=problem_types.index, autopct='%1.1f%%',
               colors=['lightcoral', 'lightblue'])
        ax4.set_title('Problem Type Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rl_analysis(self, 
                        rl_results: Dict[str, Any],
                        save_path: Optional[str] = None) -> None:
        """
        Plot RL-ABC specific analysis.
        
        Args:
            rl_results: Results from ExperimentRunner.run_rl_analysis()
            save_path: Optional path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Convergence
        convergence = rl_results['convergence_history']
        ax1.plot(convergence, linewidth=2, color='green')
        ax1.set_title('RL-ABC Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Fitness')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RL Action preferences
        rl_stats = rl_results['rl_statistics']
        action_prefs = rl_stats['action_preferences']
        action_names = ['Standard\nSearch', 'Enhanced\nExploration', 'Enhanced\nExploitation']
        bars2 = ax2.bar(action_names, action_prefs, color=['skyblue', 'orange', 'lightgreen'])
        ax2.set_title('RL Action Preferences')
        ax2.set_ylabel('Average Q-Value')
        for i, pref in enumerate(action_prefs):
            ax2.text(i, pref + max(action_prefs)*0.02, f'{pref:.3f}', 
                    ha='center', fontweight='bold')
        
        # Plot 3: Q-table statistics
        q_stats = rl_stats['q_table_stats']
        stats_names = ['Mean', 'Std', 'Max', 'Min']
        stats_values = [q_stats['mean'], q_stats['std'], q_stats['max'], q_stats['min']]
        bars3 = ax3.bar(stats_names, stats_values, color='lightcoral')
        ax3.set_title('Q-Table Statistics')
        ax3.set_ylabel('Value')
        for i, val in enumerate(stats_values):
            ax3.text(i, val + max(stats_values)*0.02, f'{val:.3f}', 
                    ha='center', fontweight='bold')
        
        # Plot 4: SVM Results
        svm_results = rl_results['svm_results']
        metrics = ['CV Score', 'Test Accuracy']
        values = [svm_results['cv_score'], svm_results['test_accuracy']]
        bars4 = ax4.bar(metrics, values, color=['lightblue', 'lightgreen'])
        ax4.set_title('SVM Optimization Results')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        for i, val in enumerate(values):
            ax4.text(i, val + 0.02, f'{val:.4f}', ha='center', fontweight='bold')
        
        fig.suptitle(f'RL-ABC Analysis (ε={rl_stats["epsilon"]:.3f})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, 
                                  results: Dict[str, Any],
                                  metric: str = 'objective') -> None:
        """
        Plot performance comparison using box plots.
        
        Args:
            results: Results from ExperimentRunner.compare_algorithms()
            metric: Metric to compare ('objective', 'time', 'success_rate')
        """
        algorithms = results['algorithms']
        
        if metric == 'objective':
            data = [alg_results['objective_values'] for alg_results in algorithms.values()]
            ylabel = 'Final Objective Value'
            title = f'Performance Comparison - {results["problem"]}'
            log_scale = True
        elif metric == 'time':
            data = [alg_results['execution_times'] for alg_results in algorithms.values()]
            ylabel = 'Execution Time (seconds)'
            title = f'Runtime Comparison - {results["problem"]}'
            log_scale = False
        elif metric == 'success_rate':
            data = [[alg_results['success_rate']] * len(alg_results['objective_values']) 
                   for alg_results in algorithms.values()]
            ylabel = 'Success Rate'
            title = f'Success Rate Comparison - {results["problem"]}'
            log_scale = False
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        box_plot = plt.boxplot(data, labels=list(algorithms.keys()), patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if log_scale and metric == 'objective':
            plt.yscale('log')
        
        # Add mean values as text
        for i, (alg_name, alg_results) in enumerate(algorithms.items()):
            if metric == 'objective':
                mean_val = alg_results['mean_objective']
            elif metric == 'time':
                mean_val = alg_results['mean_time']
            elif metric == 'success_rate':
                mean_val = alg_results['success_rate']
            
            plt.text(i + 1, mean_val, f'{mean_val:.2e}' if metric == 'objective' else f'{mean_val:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_improvement_summary(self, results: Dict[str, Any]) -> None:
        """
        Plot improvement summary showing percentage improvements.
        
        Args:
            results: Results from ExperimentRunner.compare_algorithms()
        """
        comparison = results['comparison']
        improvements = comparison['improvements']
        
        alg_names = list(improvements.keys())
        obj_improvements = [improvements[alg]['objective_improvement_%'] for alg in alg_names]
        time_overheads = [improvements[alg]['time_overhead_%'] for alg in alg_names]
        
        x = np.arange(len(alg_names))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Objective improvement
        bars1 = ax1.bar(x, obj_improvements, width, 
                       color=['orange', 'red'][:len(alg_names)], alpha=0.7)
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Objective Improvement (%)')
        ax1.set_title(f'Objective Improvement vs {comparison["baseline"]}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(alg_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, obj_improvements):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Time overhead
        bars2 = ax2.bar(x, time_overheads, width, 
                       color=['orange', 'red'][:len(alg_names)], alpha=0.7)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Time Overhead (%)')
        ax2.set_title(f'Runtime Overhead vs {comparison["baseline"]}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(alg_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, time_overheads):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_parameter_sensitivity(self, results: Dict[str, Any]) -> None:
        """
        Plot parameter sensitivity analysis results.
        
        Args:
            results: Results from ExperimentRunner.parameter_sensitivity_analysis()
        """
        param_results = results['results']
        
        # Extract parameter values and performance
        param1_values = [r['parameters']['sa_max_iters'] for r in param_results]
        param2_values = [r['parameters']['sa_perturb_scale'] for r in param_results]
        performance = [r['performance'] for r in param_results]
        
        # Create pivot table for heatmap
        unique_param1 = sorted(list(set(param1_values)))
        unique_param2 = sorted(list(set(param2_values)))
        
        performance_matrix = np.zeros((len(unique_param2), len(unique_param1)))
        
        for i, p2 in enumerate(unique_param2):
            for j, p1 in enumerate(unique_param1):
                # Find performance for this parameter combination
                for r in param_results:
                    if (r['parameters']['sa_max_iters'] == p1 and 
                        r['parameters']['sa_perturb_scale'] == p2):
                        performance_matrix[i, j] = r['performance']
                        break
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(performance_matrix, 
                   xticklabels=unique_param1,
                   yticklabels=unique_param2,
                   annot=True, fmt='.2e', cmap='viridis_r')
        
        plt.xlabel('SA Max Iterations')
        plt.ylabel('SA Perturbation Scale')
        plt.title(f'Parameter Sensitivity Analysis - {results["problem"]}')
        plt.tight_layout()
        plt.show()
    
    def plot_scalability_analysis(self, results: Dict[str, Any]) -> None:
        """
        Plot scalability analysis results.
        
        Args:
            results: Results from ExperimentRunner.scalability_analysis()
        """
        algorithms = results['algorithms']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        for alg_name, alg_results in algorithms.items():
            dimensions = [r['dimensions'] for r in alg_results]
            mean_objectives = [r['mean_objective'] for r in alg_results]
            mean_times = [r['mean_time'] for r in alg_results]
            success_rates = [r['success_rate'] for r in alg_results]
            std_objectives = [r['std_objective'] for r in alg_results]
            
            # Plot mean objective vs dimensions
            ax1.plot(dimensions, mean_objectives, 'o-', label=alg_name, linewidth=2)
            ax1.set_xlabel('Dimensions')
            ax1.set_ylabel('Mean Objective Value')
            ax1.set_title('Objective vs Problem Size')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot execution time vs dimensions
            ax2.plot(dimensions, mean_times, 's-', label=alg_name, linewidth=2)
            ax2.set_xlabel('Dimensions')
            ax2.set_ylabel('Mean Execution Time (s)')
            ax2.set_title('Runtime vs Problem Size')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot success rate vs dimensions
            ax3.plot(dimensions, success_rates, '^-', label=alg_name, linewidth=2)
            ax3.set_xlabel('Dimensions')
            ax3.set_ylabel('Success Rate')
            ax3.set_title('Success Rate vs Problem Size')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Plot standard deviation vs dimensions
            ax4.plot(dimensions, std_objectives, 'd-', label=alg_name, linewidth=2)
            ax4.set_xlabel('Dimensions')
            ax4.set_ylabel('Std Objective Value')
            ax4.set_title('Solution Robustness vs Problem Size')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.suptitle(f'Scalability Analysis - {results["problem_name"]}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a formatted summary of results.
        
        Args:
            results: Results from ExperimentRunner.compare_algorithms()
        """
        print("=" * 80)
        print(f"EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)
        print(f"Problem: {results['problem']}")
        print(f"Dimensions: {results['dimensions']}")
        print(f"Max Iterations: {results['max_iterations']}")
        print(f"Population Size: {results['num_bees']}")
        print(f"Number of Runs: {results['num_runs']}")
        print()
        
        algorithms = results['algorithms']
        
        # Performance table
        print(f"{'Algorithm':<20} {'Mean Obj':<12} {'Std Obj':<12} {'Best Obj':<12} {'Mean Time':<12} {'Success Rate':<12}")
        print("-" * 92)
        
        for alg_name, alg_results in algorithms.items():
            print(f"{alg_name:<20} "
                  f"{alg_results['mean_objective']:<12.2e} "
                  f"{alg_results['std_objective']:<12.2e} "
                  f"{alg_results['best_objective']:<12.2e} "
                  f"{alg_results['mean_time']:<12.3f} "
                  f"{alg_results['success_rate']:<12.3f}")
        
        print()
        
        # Improvements
        if 'comparison' in results:
            comparison = results['comparison']
            print(f"IMPROVEMENTS vs {comparison['baseline']}:")
            print("-" * 50)
            
            for alg_name, improvements in comparison['improvements'].items():
                print(f"{alg_name}:")
                print(f"  Objective improvement: {improvements['objective_improvement_%']:.2f}%")
                print(f"  Time overhead: {improvements['time_overhead_%']:.2f}%")
                print(f"  Success rate improvement: {improvements['success_rate_improvement']:.3f}")
                print()
        
        print("=" * 80)