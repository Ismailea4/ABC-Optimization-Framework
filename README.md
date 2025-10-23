# 🐝 ABC Optimization Framework

A comprehensive and extensible framework for the **Artificial Bee Colony (ABC)** algorithm and its hybrid variants, now featuring **Machine Learning hyperparameter optimization** capabilities.

This project provides a clean, modular structure for experimenting with **metaheuristic enhancements**, including:
- 🔥 **Simulated Annealing (SA)** integration for local search refinement
- 🌟 **Population-Based Simulated Annealing (PBSA)** for enhanced global exploration
- 🎯 **SVM Hyperparameter Optimization** for machine learning applications
- 📊 **Multi-dataset benchmarking** and performance comparison tools
- 🔄 **Adaptive limits** and exploration strategies
- 🏆 **Elitism** and hybrid swarm intelligence

---

## 🚀 Key Features

### Core Algorithm Features
- **Enhanced ABC Algorithm**: Advanced implementation with SA and PBSA integration
- **Basic ABC Algorithm**: Traditional implementation for comparison studies
- **Modular Design**: Easy to extend with new optimization strategies
- **Real-time Monitoring**: Convergence tracking and performance visualization

### Machine Learning Integration
- **SVM Optimization**: Automated hyperparameter tuning for Support Vector Machines
- **Multi-kernel Support**: RBF, Linear, Polynomial, and Sigmoid kernels
- **Cross-validation**: Robust performance evaluation with k-fold CV
- **Multi-dataset Testing**: Built-in support for popular ML datasets
- **Performance Comparison**: Baseline vs. optimized model evaluation

### Benchmarking & Analysis
- **Classical Benchmark Functions**: Sphere, Rastrigin, Ackley, Rosenbrock
- **Statistical Analysis**: Multi-run experiments with statistical significance
- **Visualization Tools**: Convergence plots, performance comparisons
- **Scalability Testing**: Performance analysis across different problem dimensions

---

## 📦 Installation

### Quick Setup
```bash
git clone https://github.com/<your-username>/ABC-Optimization-Framework.git
cd ABC-Optimization-Framework
pip install -r requirements.txt
```

### Development Installation
```bash
git clone https://github.com/<your-username>/ABC-Optimization-Framework.git
cd ABC-Optimization-Framework
pip install -e .
```

### Requirements
- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0 (for ML features)
- Seaborn >= 0.11.0 (optional, for enhanced visualization)

---

## 🎯 Quick Start

### 1. Basic ABC Optimization
```python
from abc_framework import EnhancedABC
from abc_framework.problems.benchmark_functions import sphere_function
import numpy as np

# Define problem
def objective(x):
    return sphere_function(x)

# Set up bounds (10-dimensional problem)
bounds = [(-5.0, 5.0)] * 10

# Create and run ABC algorithm
abc = EnhancedABC(
    num_bees=30,
    bounds=bounds,
    objective_function=objective,
    max_iterations=100
)

best_solution, best_fitness = abc.optimize()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

### 2. SVM Hyperparameter Optimization
```python
from abc_framework.problems.svm_optimization import create_svm_problem
from abc_framework import EnhancedABC

# Create SVM optimization problem
svm_problem = create_svm_problem(
    dataset_name='iris',
    kernel='rbf',
    optimize_params=['C', 'gamma']
)

# Run optimization
abc = EnhancedABC(
    num_bees=25,
    bounds=svm_problem.bounds,
    objective_function=svm_problem.evaluate,
    max_iterations=50
)

best_solution, best_fitness = abc.optimize()

# Evaluate best model
results = svm_problem.evaluate_best_model()
print(f"Best parameters: {results['best_params']}")
print(f"Test accuracy: {results['test_accuracy']:.4f}")
```

### 3. Complete SVM Experiment
```python
from examples.svm_examples import SVMExperimentRunner

runner = SVMExperimentRunner()

# Run comprehensive experiment
result = runner.run_single_experiment(
    dataset_name='wine',
    algorithm_type='enhanced',
    kernel='rbf',
    num_bees=30,
    max_iterations=50
)

print(f"Improvement: {result['improvement_percentage']:.2f}%")
```

---

## 📚 Usage Guide

### Algorithm Types

#### Enhanced ABC (Recommended)
- Includes Simulated Annealing local search
- Population-Based Simulated Annealing refinement
- Better convergence and solution quality

```python
abc = EnhancedABC(
    num_bees=30,
    bounds=bounds,
    objective_function=objective,
    max_iterations=100,
    use_sa=True,      # Enable Simulated Annealing
    use_pbsa=True,    # Enable Population-Based SA
    pbsa_interval=10  # Apply PBSA every 10 iterations
)
```

#### Basic ABC
- Traditional ABC implementation
- Useful for comparison studies

```python
abc = EnhancedABC(
    num_bees=30,
    bounds=bounds,
    objective_function=objective,
    max_iterations=100,
    use_sa=False,     # Disable enhancements
    use_pbsa=False
)
```

### SVM Optimization Guide

#### Supported Datasets
- **Classification**: iris, wine, breast_cancer
- **Regression**: diabetes, boston (if available)

#### Supported Kernels
- **RBF**: Optimizes C and gamma parameters
- **Linear**: Optimizes C parameter only
- **Polynomial**: Optimizes C, gamma, and degree
- **Sigmoid**: Optimizes C and gamma parameters

#### Parameter Optimization
```python
# Example: Optimize different parameter combinations
svm_problem = create_svm_problem(
    dataset_name='breast_cancer',
    kernel='rbf',
    optimize_params=['C', 'gamma'],  # Specify parameters to optimize
    cv_folds=5,                      # Cross-validation folds
    test_size=0.2                    # Test set proportion
)
```

#### Performance Comparison
```python
runner = SVMExperimentRunner()

# Compare algorithms
comparison = runner.compare_algorithms(
    dataset_name='iris',
    kernels=['rbf', 'linear'],
    algorithms=['basic', 'enhanced'],
    num_runs=5
)

# Compare across datasets
dataset_comparison = runner.run_dataset_comparison(
    datasets=['iris', 'wine', 'breast_cancer'],
    algorithm='enhanced'
)
```

---

## 🔬 Examples and Experiments

### Running Examples
```bash
# Run all examples
python examples.py

# Run specific SVM examples
python examples/svm_examples.py
```

### Available Examples

1. **Basic Optimization**: Simple sphere function optimization
2. **Algorithm Comparison**: Enhanced vs Basic ABC performance
3. **Parameter Sensitivity**: Impact of algorithm parameters
4. **Scalability Analysis**: Performance across problem dimensions
5. **SVM Optimization**: ML hyperparameter tuning examples
6. **Multi-dataset Comparison**: Performance across different datasets

---

## 📊 Project Structure

```
ABC-Optimization-Framework/
├── core/                          # Core algorithm implementations
│   ├── base_abc.py               # Abstract base ABC class
│   ├── enhanced_abc.py           # Enhanced ABC with SA/PBSA
│   ├── population.py             # Population management
│   ├── local_search.py           # SA and PBSA implementations
│   └── utils.py                  # Utility functions
├── problems/                      # Optimization problems
│   ├── benchmark_functions.py    # Classical test functions
│   ├── custom_problem.py         # Problem base classes
│   └── svm_optimization.py       # SVM hyperparameter optimization
├── examples/                      # Example implementations
│   └── svm_examples.py           # SVM optimization examples
├── experiments/                   # Experimental tools
│   ├── run_experiment.py         # Experiment runner
│   └── visualize_results.py      # Visualization tools
├── examples.py                    # Main examples file
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## 🎨 Visualization and Analysis

### Convergence Plots
```python
# Plot convergence comparison
from examples.svm_examples import plot_convergence_comparison

results = [result1, result2, result3]  # From experiments
plot_convergence_comparison(results, save_path='convergence.png')
```

### Performance Analysis
```python
# Statistical comparison
runner = SVMExperimentRunner()
df = runner.compare_algorithms(
    dataset_name='wine',
    algorithms=['basic', 'enhanced'],
    num_runs=10
)
print(df)  # Formatted results table
```

---

## 🚀 Advanced Usage

### Custom Problem Definition
```python
from abc_framework.problems.custom_problem import OptimizationProblem
import numpy as np

class MyCustomProblem(OptimizationProblem):
    def __init__(self):
        bounds = [(-10, 10)] * 5  # 5-dimensional problem
        super().__init__(dimensions=5, bounds=bounds)
    
    def evaluate(self, x):
        # Define your objective function
        return np.sum(x**2) + np.sin(np.sum(x))

# Use with ABC
problem = MyCustomProblem()
abc = EnhancedABC(
    num_bees=30,
    bounds=problem.bounds,
    objective_function=problem.evaluate,
    max_iterations=100
)
```

### Custom SVM Optimization
```python
from abc_framework.problems.svm_optimization import SVMOptimizationProblem
from sklearn.datasets import make_classification

# Create custom dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Create optimization problem
svm_problem = SVMOptimizationProblem(
    X=X, y=y,
    problem_type='classification',
    kernel_type='rbf',
    optimize_params=['C', 'gamma'],
    cv_folds=5
)

# Optimize
abc = EnhancedABC(
    num_bees=30,
    bounds=svm_problem.bounds,
    objective_function=svm_problem.evaluate,
    max_iterations=50
)
best_solution, best_fitness = abc.optimize()
```

---

## 📈 Performance Tips

### Algorithm Configuration
- **Small problems** (< 10 dimensions): 20-30 bees, 50-100 iterations
- **Medium problems** (10-50 dimensions): 30-50 bees, 100-200 iterations  
- **Large problems** (> 50 dimensions): 50-100 bees, 200-500 iterations

### SVM Optimization Tips
- Use **RBF kernel** for most problems (good default choice)
- Start with **25-30 bees** and **40-60 iterations** for quick results
- For final results, use **50+ bees** and **100+ iterations**
- Enable both **SA and PBSA** for best performance
- Use **5-fold CV** for reliable performance estimates

### Memory and Speed Optimization
- Use smaller population sizes for quick experiments
- Reduce CV folds for faster feedback during development
- Enable multiprocessing in scikit-learn (automatically used)
- Monitor convergence - stop early if no improvement

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all public functions
- Include unit tests for new features
- Update documentation for API changes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Bio-inspired Algorithms Course
- Artificial Bee Colony algorithm by Dervis Karaboga
- Scikit-learn for machine learning utilities
- Open source community for inspiration and support

---

## 🧪 Running the Examples

### Quick Test
```bash
# Test the SVM optimization functionality
python test_svm.py

# Run simple SVM optimization examples
python simple_svm_example.py

# Compare Basic vs Enhanced ABC
python comparison_example.py

# Run all framework examples
python examples.py
```

### Expected Results
- **SVM Optimization**: Typical improvements of 1-5% in accuracy
- **Convergence**: Usually converges within 30-50 iterations
- **Time**: 30-300 seconds depending on dataset size and configuration

---

## � Troubleshooting

### Common Issues

**Problem**: ImportError for scikit-learn
```bash
# Solution
pip install scikit-learn>=1.0.0
```

**Problem**: Poor optimization results
```python
# Try these solutions:
# 1. Increase population size and iterations
abc = EnhancedABC(num_bees=50, max_iterations=100, ...)

# 2. Enable all enhancement features
abc = EnhancedABC(use_sa=True, use_pbsa=True, ...)

# 3. Reduce cross-validation folds for quicker feedback
svm_problem = create_svm_problem(cv_folds=3, ...)
```

**Problem**: Slow performance
```python
# Speed optimization tips:
# 1. Start with smaller configurations
abc = EnhancedABC(num_bees=15, max_iterations=20, ...)

# 2. Use fewer CV folds
svm_problem = create_svm_problem(cv_folds=3, ...)

# 3. Test on smaller datasets first (iris, wine)
```

---

## �📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation and examples
- Run the test suite: `python test_svm.py`

### Getting Help
1. **First**: Run `python test_svm.py` to verify installation
2. **Try**: Simple examples in `simple_svm_example.py`
3. **Check**: The Quick Start guide in `QUICKSTART.md`
4. **Compare**: Algorithm performance with `comparison_example.py`

Happy optimizing! 🐝✨
