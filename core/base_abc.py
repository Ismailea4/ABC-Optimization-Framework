"""
Base ABC algorithm implementation
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Dict, Any
from .population import Population
from .utils import roulette_wheel_selection


class BaseABC(ABC):
    """
    Abstract base class for ABC algorithms.
    """
    
    def __init__(self, 
                 num_bees: int,
                 bounds: List[Tuple[float, float]],
                 objective_function: Callable[[np.ndarray], float],
                 max_iterations: int = 100,
                 limit: int = 20,
                 is_minimize: bool = True):
        """
        Initialize the ABC algorithm.
        
        Args:
            num_bees: Number of employed/onlooker bees
            bounds: Parameter bounds for each dimension
            objective_function: Function to optimize
            max_iterations: Maximum number of iterations
            limit: Trial limit for scout phase
            is_minimize: Whether this is a minimization problem
        """
        self.num_bees = num_bees
        self.bounds = bounds
        self.objective_function = objective_function
        self.max_iterations = max_iterations
        self.limit = limit
        self.is_minimize = is_minimize
        self.dimensions = len(bounds)
        
        # Initialize population
        self.population = Population(num_bees, bounds, objective_function, is_minimize)
        
        # History tracking
        self.convergence_history = []
        self.best_fitness_history = []
        self.iteration = 0
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the optimization algorithm.
        
        Returns:
            Tuple of (best_solution, best_objective_value)
        """
        self.convergence_history = []
        self.best_fitness_history = []
        
        for self.iteration in range(self.max_iterations):
            # Main ABC phases
            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()
            
            # Custom processing for subclasses
            self._post_iteration_processing()
            
            # Record convergence
            self.convergence_history.append(self.population.best_objective)
            self.best_fitness_history.append(self.population.best_fitness)
        
        return self.population.best_solution, self.population.best_objective
    
    def _employed_bee_phase(self):
        """Employed bee phase - explore neighborhood of each solution."""
        for i in range(self.num_bees):
            # Generate neighbor solution
            new_solution = self.population.get_neighbor_solution(i)
            
            # Apply custom enhancements if available
            new_solution = self._enhance_solution(new_solution, i)
            
            # Update population
            self.population.update_solution(i, new_solution)
    
    def _onlooker_bee_phase(self):
        """Onlooker bee phase - probabilistic selection and exploitation."""
        for _ in range(self.num_bees):
            # Select solution based on fitness (roulette wheel)
            selected_index = roulette_wheel_selection(self.population.fitness_values)
            
            # Generate neighbor solution
            new_solution = self.population.get_neighbor_solution(selected_index)
            
            # Apply custom enhancements if available
            new_solution = self._enhance_solution(new_solution, selected_index)
            
            # Update population
            self.population.update_solution(selected_index, new_solution)
    
    def _scout_bee_phase(self):
        """Scout bee phase - replace exhausted solutions."""
        self.population.scout_search(self.limit)
    
    @abstractmethod
    def _enhance_solution(self, solution: np.ndarray, index: int) -> np.ndarray:
        """
        Abstract method for solution enhancement (e.g., local search).
        
        Args:
            solution: Solution to enhance
            index: Index of the solution in population
        
        Returns:
            Enhanced solution
        """
        pass
    
    def _post_iteration_processing(self):
        """
        Hook for additional processing after each iteration.
        Override in subclasses for custom behavior.
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dictionary with optimization statistics
        """
        stats = self.population.get_statistics()
        stats.update({
            'iterations_completed': self.iteration + 1,
            'convergence_history': self.convergence_history.copy(),
            'best_fitness_history': self.best_fitness_history.copy()
        })
        return stats


class StandardABC(BaseABC):
    """
    Standard ABC algorithm without enhancements.
    """
    
    def _enhance_solution(self, solution: np.ndarray, index: int) -> np.ndarray:
        """No enhancement for standard ABC."""
        return solution