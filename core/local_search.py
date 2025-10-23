"""
Local search methods for ABC optimization
"""
import numpy as np
from typing import List, Tuple, Callable, Dict, Any, Optional
from .utils import clamp_solution_to_bounds


class SimulatedAnnealing:
    """
    Simulated Annealing local search for solution refinement.
    """
    
    def __init__(self, 
                 max_iters: int = 10,
                 initial_temperature: float = 0.5,
                 cooling_rate: float = 0.95,
                 perturbation_scale: float = 0.02):
        """
        Initialize SA parameters.
        
        Args:
            max_iters: Maximum number of SA iterations
            initial_temperature: Initial temperature
            cooling_rate: Temperature cooling rate (alpha)
            perturbation_scale: Scale of perturbations
        """
        self.max_iters = max_iters
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.perturbation_scale = perturbation_scale
    
    def optimize(self, 
                 initial_solution: np.ndarray,
                 fitness_function: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Perform simulated annealing optimization.
        
        Args:
            initial_solution: Starting solution
            fitness_function: Function to evaluate fitness (higher is better)
            bounds: Parameter bounds
        
        Returns:
            Improved solution
        """
        current_solution = np.copy(initial_solution)
        current_fitness = fitness_function(current_solution)
        best_solution = np.copy(current_solution)
        best_fitness = current_fitness
        
        temperature = self.initial_temperature
        
        for iteration in range(self.max_iters):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution, bounds)
            neighbor_fitness = fitness_function(neighbor)
            
            # Accept or reject based on SA criteria
            delta_fitness = neighbor_fitness - current_fitness
            
            if (delta_fitness > 0 or 
                (temperature > 0 and np.random.random() < np.exp(delta_fitness / temperature))):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                # Update best if improved
                if current_fitness > best_fitness:
                    best_solution = np.copy(current_solution)
                    best_fitness = current_fitness
            
            # Cool down
            temperature *= self.cooling_rate
        
        return best_solution
    
    def _generate_neighbor(self, 
                          solution: np.ndarray, 
                          bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Generate a neighbor solution by perturbing random dimensions.
        
        Args:
            solution: Current solution
            bounds: Parameter bounds
        
        Returns:
            Neighbor solution
        """
        neighbor = np.copy(solution)
        
        # Perturb 1-3 random dimensions
        num_perturb = np.random.randint(1, min(4, len(solution) + 1))
        dims_to_perturb = np.random.choice(len(solution), size=num_perturb, replace=False)
        
        for dim in dims_to_perturb:
            min_val, max_val = bounds[dim]
            range_val = max_val - min_val
            perturbation = np.random.normal(0, self.perturbation_scale * range_val)
            neighbor[dim] = solution[dim] + perturbation
        
        # Clamp to bounds
        return clamp_solution_to_bounds(neighbor, bounds)


class PopulationBasedSA:
    """
    Population-Based Simulated Annealing (PBSA) refinement.
    """
    
    def __init__(self, 
                 elite_percentage: float = 0.2,
                 sa_params: Optional[Dict[str, Any]] = None):
        """
        Initialize PBSA parameters.
        
        Args:
            elite_percentage: Percentage of population to refine
            sa_params: Parameters for SA refinement
        """
        self.elite_percentage = elite_percentage
        
        # Default SA parameters for PBSA
        default_sa_params = {
            'max_iters': 15,
            'initial_temperature': 0.3,
            'cooling_rate': 0.9,
            'perturbation_scale': 0.01
        }
        
        if sa_params:
            default_sa_params.update(sa_params)
        
        self.sa = SimulatedAnnealing(**default_sa_params)
    
    def refine_population(self, 
                         population: List[np.ndarray],
                         fitness_function: Callable[[np.ndarray], float],
                         bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
        """
        Apply PBSA refinement to a population.
        
        Args:
            population: List of solution vectors
            fitness_function: Function to evaluate fitness
            bounds: Parameter bounds
        
        Returns:
            Refined population
        """
        pop_size = len(population)
        elite_size = max(1, int(self.elite_percentage * pop_size))
        
        # Evaluate fitness for all individuals
        fitness_values = [fitness_function(individual) for individual in population]
        
        # Get indices sorted by fitness (descending - best first)
        sorted_indices = sorted(range(pop_size), 
                              key=lambda i: fitness_values[i], 
                              reverse=True)
        
        # Select elites and worst individuals
        elite_indices = sorted_indices[:elite_size]
        worst_indices = sorted_indices[-elite_size:]
        
        # Refine elites with SA
        refined_elites = []
        for idx in elite_indices:
            refined_solution = self.sa.optimize(
                population[idx], 
                fitness_function, 
                bounds
            )
            refined_elites.append(refined_solution)
        
        # Create new population by replacing worst with refined elites
        new_population = population.copy()
        for i, worst_idx in enumerate(worst_indices):
            if i < len(refined_elites):
                new_population[worst_idx] = refined_elites[i]
        
        return new_population


class LocalSearchManager:
    """
    Manager for coordinating different local search methods.
    """
    
    def __init__(self):
        self.sa = None
        self.pbsa = None
        self.sa_calls = 0
        self.pbsa_calls = 0
    
    def setup_sa(self, sa_params: Dict[str, Any]):
        """Setup Simulated Annealing."""
        self.sa = SimulatedAnnealing(**sa_params)
    
    def setup_pbsa(self, pbsa_params: Dict[str, Any]):
        """Setup Population-Based SA."""
        self.pbsa = PopulationBasedSA(**pbsa_params)
    
    def apply_sa(self, 
                 solution: np.ndarray,
                 fitness_function: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Apply SA to a single solution."""
        if self.sa is None:
            return solution
        
        self.sa_calls += 1
        return self.sa.optimize(solution, fitness_function, bounds)
    
    def apply_pbsa(self, 
                   population: List[np.ndarray],
                   fitness_function: Callable[[np.ndarray], float],
                   bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Apply PBSA to a population."""
        if self.pbsa is None:
            return population
        
        self.pbsa_calls += 1
        return self.pbsa.refine_population(population, fitness_function, bounds)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get local search statistics."""
        return {
            'sa_calls': self.sa_calls,
            'pbsa_calls': self.pbsa_calls
        }