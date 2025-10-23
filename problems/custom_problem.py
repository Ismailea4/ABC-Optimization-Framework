"""
Custom optimization problems
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Any


class OptimizationProblem(ABC):
    """
    Abstract base class for optimization problems.
    """
    
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]]):
        """
        Initialize optimization problem.
        
        Args:
            dimensions: Problem dimensionality
            bounds: Parameter bounds for each dimension
        """
        self.dimensions = dimensions
        self.bounds = bounds
    
    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function.
        
        Args:
            x: Input vector
        
        Returns:
            Objective function value
        """
        pass
    
    def __call__(self, x: np.ndarray) -> float:
        """
        Make the problem callable.
        
        Args:
            x: Input vector
        
        Returns:
            Objective function value
        """
        return self.evaluate(x)
    
    def is_feasible(self, x: np.ndarray) -> bool:
        """
        Check if a solution is feasible.
        
        Args:
            x: Solution vector
        
        Returns:
            True if feasible, False otherwise
        """
        if len(x) != self.dimensions:
            return False
        
        for i, (min_val, max_val) in enumerate(self.bounds):
            if not (min_val <= x[i] <= max_val):
                return False
        
        return True
    
    def get_random_solution(self) -> np.ndarray:
        """
        Generate a random feasible solution.
        
        Returns:
            Random solution vector
        """
        solution = np.zeros(self.dimensions)
        for i, (min_val, max_val) in enumerate(self.bounds):
            solution[i] = min_val + np.random.random() * (max_val - min_val)
        return solution


class ConstrainedProblem(OptimizationProblem):
    """
    Optimization problem with constraints.
    """
    
    def __init__(self, 
                 dimensions: int, 
                 bounds: List[Tuple[float, float]],
                 constraints: List[Callable[[np.ndarray], float]],
                 penalty_factor: float = 1e6):
        """
        Initialize constrained optimization problem.
        
        Args:
            dimensions: Problem dimensionality
            bounds: Parameter bounds for each dimension
            constraints: List of constraint functions (should be <= 0 for feasibility)
            penalty_factor: Penalty factor for constraint violations
        """
        super().__init__(dimensions, bounds)
        self.constraints = constraints
        self.penalty_factor = penalty_factor
    
    @abstractmethod
    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function without constraints.
        
        Args:
            x: Input vector
        
        Returns:
            Objective function value
        """
        pass
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate with penalty for constraint violations.
        
        Args:
            x: Input vector
        
        Returns:
            Penalized objective function value
        """
        obj_value = self.objective(x)
        
        # Add penalties for constraint violations
        penalty = 0.0
        for constraint in self.constraints:
            violation = max(0.0, constraint(x))
            penalty += violation
        
        return obj_value + self.penalty_factor * penalty
    
    def is_feasible(self, x: np.ndarray) -> bool:
        """
        Check if solution satisfies all constraints.
        
        Args:
            x: Solution vector
        
        Returns:
            True if feasible, False otherwise
        """
        if not super().is_feasible(x):
            return False
        
        for constraint in self.constraints:
            if constraint(x) > 0:
                return False
        
        return True


class MultiObjectiveProblem(OptimizationProblem):
    """
    Multi-objective optimization problem.
    """
    
    def __init__(self, 
                 dimensions: int, 
                 bounds: List[Tuple[float, float]],
                 objectives: List[Callable[[np.ndarray], float]],
                 weights: List[float] = None):
        """
        Initialize multi-objective problem.
        
        Args:
            dimensions: Problem dimensionality
            bounds: Parameter bounds for each dimension
            objectives: List of objective functions
            weights: Weights for scalarization (if None, equal weights)
        """
        super().__init__(dimensions, bounds)
        self.objectives = objectives
        self.num_objectives = len(objectives)
        
        if weights is None:
            self.weights = [1.0 / self.num_objectives] * self.num_objectives
        else:
            self.weights = weights
    
    def evaluate_objectives(self, x: np.ndarray) -> List[float]:
        """
        Evaluate all objectives separately.
        
        Args:
            x: Input vector
        
        Returns:
            List of objective values
        """
        return [obj(x) for obj in self.objectives]
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate weighted sum of objectives.
        
        Args:
            x: Input vector
        
        Returns:
            Weighted objective value
        """
        obj_values = self.evaluate_objectives(x)
        return sum(w * obj for w, obj in zip(self.weights, obj_values))


# Example custom problems

class QuadraticProblem(OptimizationProblem):
    """
    Simple quadratic optimization problem.
    """
    
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]], 
                 center: np.ndarray = None):
        """
        Initialize quadratic problem.
        
        Args:
            dimensions: Problem dimensionality
            bounds: Parameter bounds
            center: Center point for the quadratic (default: origin)
        """
        super().__init__(dimensions, bounds)
        self.center = center if center is not None else np.zeros(dimensions)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate quadratic function.
        
        Args:
            x: Input vector
        
        Returns:
            Function value
        """
        diff = x - self.center
        return np.sum(diff**2)


class SinusoidalProblem(OptimizationProblem):
    """
    Sinusoidal optimization problem with multiple local optima.
    """
    
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]], 
                 frequency: float = 1.0):
        """
        Initialize sinusoidal problem.
        
        Args:
            dimensions: Problem dimensionality
            bounds: Parameter bounds
            frequency: Frequency of oscillations
        """
        super().__init__(dimensions, bounds)
        self.frequency = frequency
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate sinusoidal function.
        
        Args:
            x: Input vector
        
        Returns:
            Function value
        """
        return np.sum(x**2) + np.sum(np.sin(self.frequency * x))


class ConstrainedQuadratic(ConstrainedProblem):
    """
    Constrained quadratic optimization example.
    """
    
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]]):
        """
        Initialize constrained quadratic problem.
        
        Args:
            dimensions: Problem dimensionality
            bounds: Parameter bounds
        """
        # Example constraint: sum(x_i) <= dimensions/2
        constraints = [lambda x: np.sum(x) - dimensions/2]
        super().__init__(dimensions, bounds, constraints)
    
    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate quadratic objective.
        
        Args:
            x: Input vector
        
        Returns:
            Objective value
        """
        return np.sum(x**2)