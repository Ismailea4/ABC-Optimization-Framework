"""
Reinforcement Learning enhanced ABC algorithm
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .enhanced_abc import EnhancedABC
import random


class RLAgent:
    """
    Simple Q-learning agent for ABC parameter adaptation.
    """
    
    def __init__(self, 
                 state_size: int = 4,
                 action_size: int = 3,
                 learning_rate: float = 0.1,
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize RL agent.
        
        Args:
            state_size: Number of state features
            action_size: Number of possible actions
            learning_rate: Q-learning rate
            epsilon: Exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: states x actions
        self.q_table = np.zeros((100, action_size))  # Discretized state space
        
    def get_state_index(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete index."""
        # Simple discretization - normalize and map to bins
        normalized_state = np.clip(state, 0, 1)
        # Combine state features into single index
        index = int(np.sum(normalized_state * [25, 25, 25, 25]) % 100)
        return index
    
    def get_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        state_idx = self.get_state_index(state)
        
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        return np.argmax(self.q_table[state_idx])
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Update Q-table using Q-learning update rule."""
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)
        
        current_q = self.q_table[state_idx, action]
        next_max_q = np.max(self.q_table[next_state_idx])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + 0.9 * next_max_q - current_q)
        self.q_table[state_idx, action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class RLABC(EnhancedABC):
    """
    Reinforcement Learning enhanced ABC algorithm.
    
    Uses RL agent to adaptively select between different search strategies:
    - Action 0: Standard ABC search
    - Action 1: Enhanced exploration (larger search radius)
    - Action 2: Enhanced exploitation (smaller search radius)
    """
    
    def __init__(self,
                 num_bees: int,
                 bounds: List[Tuple[float, float]],
                 objective_function,
                 max_iterations: int = 100,
                 limit: int = 20,
                 is_minimize: bool = True,
                 use_sa: bool = True,
                 sa_params: Optional[Dict[str, Any]] = None,
                 use_pbsa: bool = True,
                 pbsa_interval: int = 10,
                 pbsa_params: Optional[Dict[str, Any]] = None,
                 rl_learning_rate: float = 0.1,
                 rl_epsilon: float = 0.3):
        """
        Initialize RL-enhanced ABC algorithm.
        
        Args:
            rl_learning_rate: Learning rate for RL agent
            rl_epsilon: Initial exploration rate for RL agent
            Other parameters same as EnhancedABC
        """
        super().__init__(
            num_bees=num_bees,
            bounds=bounds,
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
        
        # Initialize RL agent
        self.rl_agent = RLAgent(
            learning_rate=rl_learning_rate,
            epsilon=rl_epsilon
        )
        
        # RL state tracking
        self.prev_state = None
        self.prev_action = None
        self.prev_fitness = float('inf') if is_minimize else float('-inf')
        
        # Strategy parameters
        self.base_phi = 1.0  # Base search radius scaling
        self.exploration_factor = 2.0  # Factor for enhanced exploration
        self.exploitation_factor = 0.5  # Factor for enhanced exploitation
        
    def _get_current_state(self) -> np.ndarray:
        """
        Get current state for RL agent.
        
        State features:
        - Diversity measure (std of population fitness)
        - Best fitness improvement rate
        - Convergence rate (fitness change)
        - Exploration vs exploitation balance
        """
        if len(self.population.fitness_values) < 2:
            return np.array([0.5, 0.5, 0.5, 0.5])
        
        # Feature 1: Population diversity (normalized)
        fitness_std = np.std(self.population.fitness_values)
        max_std = abs(max(self.population.fitness_values) - min(self.population.fitness_values))
        diversity = fitness_std / (max_std + 1e-8)
        diversity = np.clip(diversity, 0, 1)
        
        # Feature 2: Improvement rate
        if len(self.best_fitness_history) >= 2:
            recent_improvement = abs(self.best_fitness_history[-1] - self.best_fitness_history[-2])
            max_improvement = abs(self.best_fitness_history[0] - self.best_fitness_history[-1]) + 1e-8
            improvement_rate = recent_improvement / max_improvement
        else:
            improvement_rate = 0.5
        improvement_rate = np.clip(improvement_rate, 0, 1)
        
        # Feature 3: Convergence progress
        progress = self.iteration / self.max_iterations
        
        # Feature 4: Current search balance (based on recent actions)
        search_balance = 0.5  # Default balanced
        
        return np.array([diversity, improvement_rate, progress, search_balance])
    
    def _calculate_reward(self, current_fitness: float) -> float:
        """Calculate reward for RL agent based on fitness improvement."""
        if self.prev_fitness is None:
            return 0.0
        
        if self.is_minimize:
            improvement = self.prev_fitness - current_fitness
        else:
            improvement = current_fitness - self.prev_fitness
        
        # Normalize reward
        if improvement > 0:
            reward = min(improvement * 10, 1.0)  # Cap reward at 1.0
        else:
            reward = max(improvement * 10, -1.0)  # Cap penalty at -1.0
        
        return reward
    
    def _apply_rl_strategy(self, action: int, solution: np.ndarray, k: int) -> np.ndarray:
        """
        Apply RL-selected strategy for solution update.
        
        Args:
            action: RL agent selected action
            solution: Current solution
            k: Index of current solution
        
        Returns:
            Updated solution
        """
        j = np.random.randint(0, self.num_bees)
        while j == k:
            j = np.random.randint(0, self.num_bees)
        
        phi_base = np.random.uniform(-1, 1)
        
        # Apply strategy based on action
        if action == 0:
            # Standard ABC search
            phi = phi_base * self.base_phi
        elif action == 1:
            # Enhanced exploration (larger search radius)
            phi = phi_base * self.base_phi * self.exploration_factor
        else:  # action == 2
            # Enhanced exploitation (smaller search radius, focus on best solutions)
            if np.random.random() < 0.7:  # 70% chance to use best solutions
                # Select from top 30% solutions
                top_indices = np.argsort(self.population.fitness_values)
                if self.is_minimize:
                    j = top_indices[:max(1, self.num_bees // 3)]
                else:
                    j = top_indices[-max(1, self.num_bees // 3):]
                j = np.random.choice(j)
            phi = phi_base * self.base_phi * self.exploitation_factor
        
        # Generate new solution
        dimension = np.random.randint(0, self.dimensions)
        new_solution = solution.copy()
        
        partner_solution = self.population.solutions[j]
        new_solution[dimension] = solution[dimension] + phi * (solution[dimension] - partner_solution[dimension])
        
        # Apply bounds
        new_solution = self._clamp_to_bounds(new_solution)
        
        return new_solution
    
    def _employed_bee_phase(self):
        """Enhanced employed bee phase with RL strategy selection."""
        current_state = self._get_current_state()
        
        for i in range(self.num_bees):
            # Get action from RL agent
            action = self.rl_agent.get_action(current_state)
            
            # Apply RL strategy
            new_solution = self._apply_rl_strategy(action, self.population.solutions[i], i)
            
            # Evaluate new solution
            new_fitness = self.objective_function(new_solution)
            
            # Update solution if better
            if self._is_better(new_fitness, self.population.fitness_values[i]):
                self.population.solutions[i] = new_solution
                self.population.fitness_values[i] = new_fitness
                self.population.trials[i] = 0
                
                # Calculate reward and update RL agent
                reward = self._calculate_reward(new_fitness)
                next_state = self._get_current_state()
                
                if self.prev_state is not None and self.prev_action is not None:
                    self.rl_agent.update_q_table(self.prev_state, self.prev_action, reward, next_state)
                
                self.prev_state = current_state
                self.prev_action = action
                self.prev_fitness = new_fitness
            else:
                self.population.trials[i] += 1
                
                # Negative reward for no improvement
                if self.prev_state is not None and self.prev_action is not None:
                    reward = -0.1
                    next_state = self._get_current_state()
                    self.rl_agent.update_q_table(self.prev_state, self.prev_action, reward, next_state)
    
    def _is_better(self, new_fitness: float, current_fitness: float) -> bool:
        """Check if new fitness is better than current fitness."""
        if self.is_minimize:
            return new_fitness < current_fitness
        else:
            return new_fitness > current_fitness
    
    def _onlooker_bee_phase(self):
        """Standard onlooker bee phase."""
        # Calculate selection probabilities
        fitness_values = np.array(self.population.fitness_values)
        if self.is_minimize:
            # For minimization, invert fitness values
            min_fitness = np.min(fitness_values)
            max_fitness = np.max(fitness_values)
            if max_fitness > min_fitness:
                probabilities = (max_fitness - fitness_values) / (max_fitness - min_fitness)
            else:
                probabilities = np.ones(len(fitness_values))
        else:
            # For maximization
            min_fitness = np.min(fitness_values)
            if np.max(fitness_values) > min_fitness:
                probabilities = (fitness_values - min_fitness) / (np.max(fitness_values) - min_fitness)
            else:
                probabilities = np.ones(len(fitness_values))
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Onlooker bee selection and search
        for _ in range(self.num_bees):
            # Select solution based on probability
            selected_idx = np.random.choice(self.num_bees, p=probabilities)
            
            # Apply search similar to employed bees but without RL
            j = np.random.randint(0, self.num_bees)
            while j == selected_idx:
                j = np.random.randint(0, self.num_bees)
            
            # Generate new solution
            phi = np.random.uniform(-1, 1)
            dimension = np.random.randint(0, self.dimensions)
            new_solution = self.population.solutions[selected_idx].copy()
            
            partner_solution = self.population.solutions[j]
            new_solution[dimension] = (self.population.solutions[selected_idx][dimension] + 
                                     phi * (self.population.solutions[selected_idx][dimension] - 
                                           partner_solution[dimension]))
            
            # Apply bounds
            new_solution = self._clamp_to_bounds(new_solution)
            
            # Evaluate new solution
            new_fitness = self.objective_function(new_solution)
            
            # Update if better
            if self._is_better(new_fitness, self.population.fitness_values[selected_idx]):
                self.population.solutions[selected_idx] = new_solution
                self.population.fitness_values[selected_idx] = new_fitness
                self.population.trials[selected_idx] = 0
            else:
                self.population.trials[selected_idx] += 1
    
    def _scout_bee_phase(self):
        """Standard scout bee phase."""
        for i in range(self.num_bees):
            if self.population.trials[i] >= self.limit:
                # Generate new random solution
                new_solution = np.array([
                    np.random.uniform(bounds[0], bounds[1]) 
                    for bounds in self.bounds
                ])
                
                new_fitness = self.objective_function(new_solution)
                
                self.population.solutions[i] = new_solution
                self.population.fitness_values[i] = new_fitness
                self.population.trials[i] = 0
    
    def _apply_sa_enhancement(self):
        """Apply SA enhancement if enabled."""
        if self.use_sa and hasattr(self, 'local_search') and self.local_search.sa:
            best_idx = np.argmin(self.population.fitness_values) if self.is_minimize else np.argmax(self.population.fitness_values)
            
            # Convert fitness to objective for SA (SA expects higher = better)
            def sa_fitness_func(solution):
                obj_val = self.objective_function(solution)
                if self.is_minimize:
                    return -obj_val  # Convert minimization to maximization
                else:
                    return obj_val
            
            enhanced_solution = self.local_search.apply_sa(
                self.population.solutions[best_idx],
                sa_fitness_func,
                self.bounds
            )
            
            enhanced_fitness = self.objective_function(enhanced_solution)
            
            if self._is_better(enhanced_fitness, self.population.fitness_values[best_idx]):
                self.population.solutions[best_idx] = enhanced_solution
                self.population.fitness_values[best_idx] = enhanced_fitness
    
    def _apply_pbsa_enhancement(self):
        """Apply PBSA enhancement if enabled."""
        if self.use_pbsa and hasattr(self, 'local_search') and self.local_search.pbsa:
            # Convert fitness to objective for PBSA
            def pbsa_fitness_func(solution):
                obj_val = self.objective_function(solution)
                if self.is_minimize:
                    return -obj_val  # Convert minimization to maximization
                else:
                    return obj_val
            
            enhanced_solutions = self.local_search.apply_pbsa(
                self.population.solutions.tolist(),
                pbsa_fitness_func,
                self.bounds
            )
            
            for i, enhanced_solution in enumerate(enhanced_solutions):
                if enhanced_solution is not None:
                    enhanced_fitness = self.objective_function(np.array(enhanced_solution))
                    
                    if self._is_better(enhanced_fitness, self.population.fitness_values[i]):
                        self.population.solutions[i] = np.array(enhanced_solution)
                        self.population.fitness_values[i] = enhanced_fitness
    
    def _clamp_to_bounds(self, solution: np.ndarray) -> np.ndarray:
        """Clamp solution to bounds."""
        clamped = solution.copy()
        for i, (lower, upper) in enumerate(self.bounds):
            clamped[i] = np.clip(clamped[i], lower, upper)
        return clamped
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run RL-enhanced ABC optimization.
        
        Returns:
            Tuple of (best_solution, best_fitness)
        """
        self.convergence_history = []
        self.best_fitness_history = []
        
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            
            # Employed bee phase with RL
            self._employed_bee_phase()
            
            # Onlooker bee phase (standard)
            self._onlooker_bee_phase()
            
            # Scout bee phase (standard)
            self._scout_bee_phase()
            
            # Apply enhancements if enabled
            if self.use_sa:
                self._apply_sa_enhancement()
            
            if self.use_pbsa and iteration % self.pbsa_interval == 0:
                self._apply_pbsa_enhancement()
            
            # Update population best
            self.population.update_best()
            
            # Update history
            best_fitness = self.population.best_fitness
            self.convergence_history.append(best_fitness)
            self.best_fitness_history.append(best_fitness)
        
        return self.population.best_solution, self.population.best_fitness
    
    def get_rl_statistics(self) -> Dict[str, Any]:
        """Get RL agent statistics."""
        return {
            'epsilon': self.rl_agent.epsilon,
            'q_table_stats': {
                'mean': np.mean(self.rl_agent.q_table),
                'std': np.std(self.rl_agent.q_table),
                'max': np.max(self.rl_agent.q_table),
                'min': np.min(self.rl_agent.q_table)
            },
            'action_preferences': np.mean(self.rl_agent.q_table, axis=0).tolist()
        }