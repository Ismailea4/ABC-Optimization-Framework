"""
ABC Framework - Advanced Artificial Bee Colony Optimization
===========================================================

A comprehensive framework for Artificial Bee Colony optimization with 
advanced features including Simulated Annealing local search and 
Population-Based Simulated Annealing (PBSA) refinement.

Author: Bio-inspired Algorithms Course
Date: October 2025
"""

from .core.base_abc import BaseABC
from .core.enhanced_abc import EnhancedABC
from .core.rl_abc import RLABC
from .core.population import Population
from .core.local_search import SimulatedAnnealing, PopulationBasedSA
from .core.utils import clamp_to_bounds, generate_random_solution

from .problems.benchmark_functions import (
    sphere_function, 
    rastrigin_function, 
    ackley_function,
    rosenbrock_function
)

from .problems.svm_optimization import (
    SVMOptimizationProblem,
    SVMDatasetLoader,
    create_svm_problem
)

from .problems.custom_problem import OptimizationProblem

__version__ = "1.0.0"
__all__ = [
    'BaseABC', 'EnhancedABC', 'RLABC', 'Population', 
    'SimulatedAnnealing', 'PopulationBasedSA',
    'sphere_function', 'rastrigin_function', 'ackley_function', 'rosenbrock_function',
    'clamp_to_bounds', 'generate_random_solution',
    'SVMOptimizationProblem', 'SVMDatasetLoader', 'create_svm_problem',
    'OptimizationProblem'
]