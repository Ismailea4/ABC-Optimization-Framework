"""
Population management for ABC optimization
"""
import numpy as np
from typing import List, Tuple, Callable
from .utils import generate_random_solution, calculate_fitness


class Population:
    """
    Manages the bee population for ABC optimization.
    """
    
    def __init__(self, 
                 size: int, 
                 bounds: List[Tuple[float, float]], 
                 objective_function: Callable[[np.ndarray], float],
                 is_minimize: bool = True):
        """
        Initialize the population.
        
        Args:
            size: Population size
            bounds: Parameter bounds for each dimension
            objective_function: Function to optimize
            is_minimize: Whether this is a minimization problem
        """
        self.size = size
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.objective_function = objective_function
        self.is_minimize = is_minimize
        
        # Initialize population
        self.solutions = np.zeros((size, self.dimensions))
        self.objective_values = np.zeros(size)
        self.fitness_values = np.zeros(size)
        self.trials = np.zeros(size)
        
        # Generate initial population
        for i in range(size):
            self.solutions[i] = generate_random_solution(bounds)
            self.objective_values[i] = objective_function(self.solutions[i])
            self.fitness_values[i] = calculate_fitness(self.objective_values[i], is_minimize)
        
        # Track best solution
        self.update_best()
    
    def update_best(self):
        """Update the best solution found so far."""
        best_index = np.argmax(self.fitness_values)
        self.best_solution = np.copy(self.solutions[best_index])
        self.best_objective = self.objective_values[best_index]
        self.best_fitness = self.fitness_values[best_index]
    
    def evaluate_solution(self, solution: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate a solution and return objective and fitness values.
        
        Args:
            solution: Solution to evaluate
        
        Returns:
            Tuple of (objective_value, fitness_value)
        """
        objective_value = self.objective_function(solution)
        fitness_value = calculate_fitness(objective_value, self.is_minimize)
        return objective_value, fitness_value
    
    def update_solution(self, index: int, new_solution: np.ndarray):
        """
        Update a solution in the population.
        
        Args:
            index: Index of solution to update
            new_solution: New solution vector
        """
        objective_value, fitness_value = self.evaluate_solution(new_solution)
        
        # Greedy selection
        if fitness_value > self.fitness_values[index]:
            self.solutions[index] = new_solution
            self.objective_values[index] = objective_value
            self.fitness_values[index] = fitness_value
            self.trials[index] = 0
            
            # Update best if necessary
            if fitness_value > self.best_fitness:
                self.best_solution = np.copy(new_solution)
                self.best_objective = objective_value
                self.best_fitness = fitness_value
        else:
            self.trials[index] += 1
    
    def scout_search(self, limit: int):
        """
        Perform scout bee search - replace exhausted sources.
        
        Args:
            limit: Maximum number of trials before abandoning a source
        """
        max_trials_index = np.argmax(self.trials)
        
        if self.trials[max_trials_index] >= limit:
            # Generate new random solution
            new_solution = generate_random_solution(self.bounds)
            self.solutions[max_trials_index] = new_solution
            
            # Evaluate new solution
            objective_value, fitness_value = self.evaluate_solution(new_solution)
            self.objective_values[max_trials_index] = objective_value
            self.fitness_values[max_trials_index] = fitness_value
            self.trials[max_trials_index] = 0
    
    def get_neighbor_solution(self, index: int) -> np.ndarray:
        """
        Generate a neighbor solution using ABC's standard method.
        
        Args:
            index: Index of the current solution
        
        Returns:
            Neighbor solution
        """
        # Select random parameter to modify
        param_index = np.random.randint(0, self.dimensions)
        
        # Select different solution for comparison
        partner_index = np.random.randint(0, self.size)
        while partner_index == index:
            partner_index = np.random.randint(0, self.size)
        
        # Create new solution
        new_solution = np.copy(self.solutions[index])
        phi = np.random.random() * 2 - 1  # [-1, 1]
        new_solution[param_index] = (new_solution[param_index] + 
                                   phi * (new_solution[param_index] - 
                                         self.solutions[partner_index, param_index]))
        
        # Apply bounds
        min_val, max_val = self.bounds[param_index]
        new_solution[param_index] = max(min(new_solution[param_index], max_val), min_val)
        
        return new_solution
    
    def get_population_copy(self) -> List[np.ndarray]:
        """
        Get a copy of the current population as a list.
        
        Returns:
            List of solution vectors
        """
        return [np.copy(solution) for solution in self.solutions]
    
    def replace_population(self, new_population: List[np.ndarray]):
        """
        Replace the current population with a new one.
        
        Args:
            new_population: List of new solution vectors
        """
        for i, solution in enumerate(new_population):
            if i < self.size:
                objective_value, fitness_value = self.evaluate_solution(solution)
                self.solutions[i] = solution
                self.objective_values[i] = objective_value
                self.fitness_values[i] = fitness_value
        
        self.update_best()
    
    def get_statistics(self) -> dict:
        """
        Get population statistics.
        
        Returns:
            Dictionary with population statistics
        """
        return {
            'best_objective': self.best_objective,
            'best_fitness': self.best_fitness,
            'mean_objective': np.mean(self.objective_values),
            'std_objective': np.std(self.objective_values),
            'mean_fitness': np.mean(self.fitness_values),
            'worst_objective': self.objective_values[np.argmin(self.fitness_values)]
        }