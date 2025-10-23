"""
Utility functions for ABC optimization framework
"""
import numpy as np
from typing import List, Tuple, Union


def clamp_to_bounds(value: float, bounds: Tuple[float, float]) -> float:
    """
    Clamp a value to the specified bounds.
    
    Args:
        value: The value to clamp
        bounds: Tuple of (min_val, max_val)
    
    Returns:
        Clamped value within bounds
    """
    min_val, max_val = bounds
    return max(min(value, max_val), min_val)


def clamp_solution_to_bounds(solution: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Clamp all parameters of a solution to their respective bounds.
    
    Args:
        solution: Solution vector to clamp
        bounds: List of (min, max) tuples for each parameter
    
    Returns:
        Clamped solution vector
    """
    clamped = np.copy(solution)
    for i, (min_val, max_val) in enumerate(bounds):
        clamped[i] = clamp_to_bounds(clamped[i], (min_val, max_val))
    return clamped


def generate_random_solution(bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Generate a random solution within the specified bounds.
    
    Args:
        bounds: List of (min, max) tuples for each parameter
    
    Returns:
        Random solution vector
    """
    solution = np.zeros(len(bounds))
    for i, (min_val, max_val) in enumerate(bounds):
        solution[i] = min_val + np.random.random() * (max_val - min_val)
    return solution


def calculate_fitness(objective_value: float, is_minimize: bool = True) -> float:
    """
    Convert objective value to fitness (higher is better).
    
    Args:
        objective_value: The objective function value
        is_minimize: Whether this is a minimization problem
    
    Returns:
        Fitness value (higher is better)
    """
    if is_minimize:
        if objective_value >= 0:
            return 1 / (1 + objective_value)
        else:
            return 1 + abs(objective_value)
    else:
        if objective_value >= 0:
            return 1 + objective_value
        else:
            return 1 / (1 + abs(objective_value))


def roulette_wheel_selection(fitness_values: np.ndarray) -> int:
    """
    Perform roulette wheel selection based on fitness values.
    
    Args:
        fitness_values: Array of fitness values
    
    Returns:
        Selected index
    """
    sum_fitness = np.sum(fitness_values)
    if sum_fitness == 0:
        return np.random.randint(0, len(fitness_values))
    
    probabilities = fitness_values / sum_fitness
    cumsum = np.cumsum(probabilities)
    r = np.random.random()
    
    for i in range(len(fitness_values)):
        if r <= cumsum[i]:
            return i
    
    return np.random.randint(0, len(fitness_values))  # Fallback


def calculate_improvement_percentage(baseline: float, improved: float) -> float:
    """
    Calculate percentage improvement.
    
    Args:
        baseline: Original value
        improved: Improved value
    
    Returns:
        Improvement percentage
    """
    if baseline == 0:
        return 0.0
    return ((baseline - improved) / abs(baseline)) * 100