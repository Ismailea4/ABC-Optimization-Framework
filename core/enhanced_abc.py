"""
Enhanced ABC algorithm with SA and PBSA
"""
import numpy as np
from typing import List, Tuple, Callable, Dict, Any, Optional
from .base_abc import BaseABC
from .local_search import LocalSearchManager


class EnhancedABC(BaseABC):
    """
    Enhanced ABC algorithm with Simulated Annealing and PBSA refinement.
    """
    
    def __init__(self, 
                 num_bees: int,
                 bounds: List[Tuple[float, float]],
                 objective_function: Callable[[np.ndarray], float],
                 max_iterations: int = 100,
                 limit: int = 20,
                 is_minimize: bool = True,
                 use_sa: bool = True,
                 sa_params: Optional[Dict[str, Any]] = None,
                 use_pbsa: bool = True,
                 pbsa_interval: int = 10,
                 pbsa_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Enhanced ABC algorithm.
        
        Args:
            num_bees: Number of employed/onlooker bees
            bounds: Parameter bounds for each dimension
            objective_function: Function to optimize
            max_iterations: Maximum number of iterations
            limit: Trial limit for scout phase
            is_minimize: Whether this is a minimization problem
            use_sa: Whether to use SA local refinement
            sa_params: Parameters for SA
            use_pbsa: Whether to use PBSA population refinement
            pbsa_interval: Interval for PBSA application
            pbsa_params: Parameters for PBSA
        """
        super().__init__(num_bees, bounds, objective_function, max_iterations, limit, is_minimize)
        
        self.use_sa = use_sa
        self.use_pbsa = use_pbsa
        self.pbsa_interval = pbsa_interval
        
        # Initialize local search manager
        self.local_search = LocalSearchManager()
        
        # Setup SA if enabled
        if use_sa:
            default_sa_params = {
                'max_iters': 10,
                'initial_temperature': 0.5,
                'cooling_rate': 0.95,
                'perturbation_scale': 0.02
            }
            if sa_params:
                default_sa_params.update(sa_params)
            self.local_search.setup_sa(default_sa_params)
        
        # Setup PBSA if enabled
        if use_pbsa:
            default_pbsa_params = {
                'elite_percentage': 0.2,
                'sa_params': {
                    'max_iters': 15,
                    'initial_temperature': 0.3,
                    'cooling_rate': 0.9,
                    'perturbation_scale': 0.01
                }
            }
            if pbsa_params:
                default_pbsa_params.update(pbsa_params)
            self.local_search.setup_pbsa(default_pbsa_params)
    
    def _enhance_solution(self, solution: np.ndarray, index: int) -> np.ndarray:
        """
        Enhance solution using SA if enabled and solution is improved.
        
        Args:
            solution: Solution to enhance
            index: Index of the solution in population
        
        Returns:
            Enhanced solution
        """
        if not self.use_sa:
            return solution
        
        # Only apply SA if the solution is better than current
        current_fitness = self.population.fitness_values[index]
        new_objective, new_fitness = self.population.evaluate_solution(solution)
        
        if new_fitness > current_fitness:
            # Apply SA refinement
            enhanced_solution = self.local_search.apply_sa(
                solution,
                lambda sol: self.population.evaluate_solution(sol)[1],  # Fitness function
                self.bounds
            )
            return enhanced_solution
        
        return solution
    
    def _post_iteration_processing(self):
        """
        Apply PBSA refinement every pbsa_interval iterations.
        """
        if (self.use_pbsa and 
            (self.iteration + 1) % self.pbsa_interval == 0):
            
            # Get current population
            current_population = self.population.get_population_copy()
            
            # Apply PBSA refinement
            refined_population = self.local_search.apply_pbsa(
                current_population,
                lambda sol: self.population.evaluate_solution(sol)[1],  # Fitness function
                self.bounds
            )
            
            # Replace population with refined version
            self.population.replace_population(refined_population)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get optimization statistics including local search info.
        
        Returns:
            Dictionary with optimization statistics
        """
        stats = super().get_statistics()
        stats.update(self.local_search.get_statistics())
        stats.update({
            'use_sa': self.use_sa,
            'use_pbsa': self.use_pbsa,
            'pbsa_interval': self.pbsa_interval
        })
        return stats


class HybridABC_SA(EnhancedABC):
    """
    Alias for backward compatibility with notebook implementation.
    """
    
    def __init__(self, 
                 num_bees: int,
                 num_parameters: int,  # For backward compatibility
                 parameter_ranges: List[Tuple[float, float]],
                 objective_function: Callable[[np.ndarray], float],
                 max_iterations: int = 100,
                 limit: int = 20,
                 is_minimize: bool = True,
                 use_sa: bool = True,
                 sa_params: Optional[Dict[str, Any]] = None,
                 use_pbsa: bool = True,
                 pbsa_interval: int = 10,
                 pbsa_params: Optional[Dict[str, Any]] = None):
        """
        Initialize with backward-compatible interface.
        """
        # Convert old parameter names
        super().__init__(
            num_bees=num_bees,
            bounds=parameter_ranges,
            objective_function=objective_function,
            max_iterations=max_iterations,
            limit=limit,
            is_minimize=is_minimize,
            use_sa=use_sa,
            sa_params=sa_params,
            use_pbsa=use_pbsa,
            pbsa_interval=pbsa_interval,
            pbsa_params=pbsa_params
        )
        
        # Backward compatibility attributes
        self.num_parameters = num_parameters
        self.parameter_ranges = parameter_ranges
        self.food_sources = self.population.solutions
        self.objective_values = self.population.objective_values
        self.fitness_values = self.population.fitness_values
        self.trials = self.population.trials
        self.best_source = self.population.best_solution
        self.best_objective = self.population.best_objective
        self.fitness_history = []
        self.sa_calls = 0
        self.pbsa_calls = 0
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run optimization with backward-compatible interface.
        
        Returns:
            Tuple of (best_solution, best_objective_value)
        """
        result = super().optimize()
        
        # Update backward compatibility attributes
        self.fitness_history = self.convergence_history.copy()
        stats = self.local_search.get_statistics()
        self.sa_calls = stats['sa_calls']
        self.pbsa_calls = stats['pbsa_calls']
        
        return result
    
    def fitness(self, solution: np.ndarray) -> float:
        """
        Backward-compatible fitness function.
        
        Args:
            solution: Solution to evaluate
        
        Returns:
            Fitness value
        """
        return self.population.evaluate_solution(solution)[1]