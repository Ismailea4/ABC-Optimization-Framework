"""
Benchmark optimization functions
"""
import numpy as np
from typing import List, Tuple


def sphere_function(x: np.ndarray) -> float:
    """
    Sphere function: f(x) = sum(x_i^2)
    
    Global minimum: f(0, ..., 0) = 0
    
    Args:
        x: Input vector
    
    Returns:
        Function value
    """
    return np.sum(x**2)


def rastrigin_function(x: np.ndarray, A: float = 10) -> float:
    """
    Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    
    Global minimum: f(0, ..., 0) = 0
    
    Args:
        x: Input vector
        A: Function parameter (default 10)
    
    Returns:
        Function value
    """
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def ackley_function(x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> float:
    """
    Ackley function: f(x) = -a*exp(-b*sqrt(1/n*sum(x_i^2))) - exp(1/n*sum(cos(c*x_i))) + a + e
    
    Global minimum: f(0, ..., 0) = 0
    
    Args:
        x: Input vector
        a: Function parameter (default 20)
        b: Function parameter (default 0.2)
        c: Function parameter (default 2*pi)
    
    Returns:
        Function value
    """
    n = len(x)
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return term1 + term2 + a + np.e


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Global minimum: f(1, ..., 1) = 0
    
    Args:
        x: Input vector
    
    Returns:
        Function value
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def griewank_function(x: np.ndarray) -> float:
    """
    Griewank function: f(x) = 1/4000 * sum(x_i^2) - prod(cos(x_i/sqrt(i))) + 1
    
    Global minimum: f(0, ..., 0) = 0
    
    Args:
        x: Input vector
    
    Returns:
        Function value
    """
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1


def schwefel_function(x: np.ndarray) -> float:
    """
    Schwefel function: f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
    
    Global minimum: f(420.9687, ..., 420.9687) ≈ 0
    
    Args:
        x: Input vector
    
    Returns:
        Function value
    """
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


# Define standard bounds for benchmark functions
BENCHMARK_BOUNDS = {
    'sphere': (-5.12, 5.12),
    'rastrigin': (-5.12, 5.12),
    'ackley': (-32.768, 32.768),
    'rosenbrock': (-5.0, 10.0),
    'griewank': (-600.0, 600.0),
    'schwefel': (-500.0, 500.0)
}


def get_benchmark_function(name: str):
    """
    Get a benchmark function by name.
    
    Args:
        name: Function name
    
    Returns:
        Function object
    
    Raises:
        ValueError: If function name is not recognized
    """
    functions = {
        'sphere': sphere_function,
        'rastrigin': rastrigin_function,
        'ackley': ackley_function,
        'rosenbrock': rosenbrock_function,
        'griewank': griewank_function,
        'schwefel': schwefel_function
    }
    
    if name.lower() not in functions:
        raise ValueError(f"Unknown function: {name}. Available: {list(functions.keys())}")
    
    return functions[name.lower()]


def get_benchmark_bounds(name: str, dimensions: int) -> List[Tuple[float, float]]:
    """
    Get standard bounds for a benchmark function.
    
    Args:
        name: Function name
        dimensions: Number of dimensions
    
    Returns:
        List of (min, max) tuples for each dimension
    
    Raises:
        ValueError: If function name is not recognized
    """
    if name.lower() not in BENCHMARK_BOUNDS:
        raise ValueError(f"Unknown function: {name}. Available: {list(BENCHMARK_BOUNDS.keys())}")
    
    bounds = BENCHMARK_BOUNDS[name.lower()]
    return [bounds] * dimensions


class BenchmarkProblem:
    """
    Wrapper class for benchmark problems.
    """
    
    def __init__(self, function_name: str, dimensions: int):
        """
        Initialize benchmark problem.
        
        Args:
            function_name: Name of the benchmark function
            dimensions: Problem dimensionality
        """
        self.function_name = function_name.lower()
        self.dimensions = dimensions
        self.function = get_benchmark_function(function_name)
        self.bounds = get_benchmark_bounds(function_name, dimensions)
        self.global_minimum = self._get_global_minimum()
    
    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the function.
        
        Args:
            x: Input vector
        
        Returns:
            Function value
        """
        return self.function(x)
    
    def _get_global_minimum(self) -> float:
        """Get the known global minimum value."""
        # Most benchmark functions have global minimum of 0
        if self.function_name in ['sphere', 'rastrigin', 'ackley', 'rosenbrock', 'griewank']:
            return 0.0
        elif self.function_name == 'schwefel':
            return 0.0  # Approximately 0 at the global optimum
        else:
            return 0.0  # Default assumption
    
    def get_error(self, solution: np.ndarray) -> float:
        """
        Calculate error from global minimum.
        
        Args:
            solution: Solution vector
        
        Returns:
            Error value
        """
        return abs(self.function(solution) - self.global_minimum)
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.function_name.title()} function ({self.dimensions}D)"