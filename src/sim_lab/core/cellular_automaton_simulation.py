"""Cellular Automaton Simulation implementation."""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


@SimulatorRegistry.register("CellularAutomaton")
class CellularAutomatonSimulation(BaseSimulation):
    """A simulation class for cellular automata.
    
    This simulation implements grid-based models where cells evolve according to
    transition rules based on the state of neighboring cells. It supports both
    Conway's Game of Life and customizable rules.
    
    Attributes:
        grid_size (Tuple[int, int]): The dimensions of the grid (rows, columns).
        initial_state (np.ndarray): Initial configuration of the grid.
        update_rule (Callable): Function that determines the next state of each cell.
        days (int): Number of generations to simulate.
        random_seed (Optional[int]): Seed for random number generation.
    """
    
    # Predefined rulesets
    GAME_OF_LIFE = "game_of_life"
    
    def __init__(
        self, 
        grid_size: Tuple[int, int] = (50, 50),
        initial_state: Optional[np.ndarray] = None,
        initial_density: float = 0.3,
        rule: Union[str, Callable] = GAME_OF_LIFE,
        days: int = 100,
        boundary: str = 'periodic',
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the cellular automaton simulation.
        
        Args:
            grid_size: The dimensions of the grid as (rows, columns).
            initial_state: Initial configuration of the grid. If None, a random grid will be generated.
            initial_density: Probability of a cell being alive in the random initial state.
            rule: Either a string for a predefined ruleset or a custom function.
            days: Number of generations to simulate.
            boundary: Boundary condition ('periodic' or 'fixed').
            random_seed: Seed for random number generation.
        """
        super().__init__(days=days, random_seed=random_seed)
        
        self.grid_size = grid_size
        self.boundary = boundary
        
        # Generate initial state if not provided
        if initial_state is not None:
            if initial_state.shape != grid_size:
                raise ValueError(f"Initial state shape {initial_state.shape} doesn't match grid size {grid_size}")
            self.initial_state = initial_state
        else:
            # Random initial state
            rng = np.random.RandomState(random_seed)
            self.initial_state = (rng.random(grid_size) < initial_density).astype(int)
        
        # Set up update rule
        if isinstance(rule, str):
            if rule == self.GAME_OF_LIFE:
                self.update_rule = self._game_of_life_rule
            else:
                raise ValueError(f"Unknown predefined rule: {rule}")
        else:
            # Custom function rule
            self.update_rule = rule
        
        # Store state history
        self.state_history = [self.initial_state.copy()]
        self.current_state = self.initial_state.copy()
    
    def _game_of_life_rule(self, grid: np.ndarray) -> np.ndarray:
        """Implementation of Conway's Game of Life rules.
        
        Rules:
        1. Any live cell with fewer than two live neighbors dies (underpopulation).
        2. Any live cell with two or three live neighbors lives on.
        3. Any live cell with more than three live neighbors dies (overpopulation).
        4. Any dead cell with exactly three live neighbors becomes a live cell (reproduction).
        
        Args:
            grid: Current state of the grid.
            
        Returns:
            Updated grid after applying the rules.
        """
        # Count live neighbors for each cell
        rows, cols = grid.shape
        new_grid = grid.copy()
        
        # Handle boundary conditions
        if self.boundary == 'periodic':
            padded = np.pad(grid, 1, mode='wrap')
        else:  # fixed boundary
            padded = np.pad(grid, 1, mode='constant', constant_values=0)
        
        # Count neighbors (including the cell itself, which we'll adjust for)
        neighbors = np.zeros((rows, cols), dtype=int)
        for i in range(3):
            for j in range(3):
                neighbors += padded[i:i+rows, j:j+cols]
        
        # Subtract the cell itself (since we counted it in the neighborhood)
        neighbors -= grid
        
        # Apply the rules
        # 1 & 3. Any live cell with <2 or >3 neighbors dies
        new_grid[(grid == 1) & ((neighbors < 2) | (neighbors > 3))] = 0
        
        # 4. Any dead cell with exactly 3 neighbors becomes alive
        new_grid[(grid == 0) & (neighbors == 3)] = 1
        
        return new_grid
    
    def run_simulation(self) -> List[float]:
        """Run the cellular automaton simulation.
        
        Evolves the grid according to the update rule for the specified number of days.
        
        Returns:
            A list with the number of live cells for each generation.
        """
        self.reset()
        
        # Initialize with the initial state
        self.current_state = self.initial_state.copy()
        self.state_history = [self.initial_state.copy()]
        
        # Run for the specified number of days
        live_cells = [np.sum(self.current_state)]
        
        for _ in range(1, self.days):
            # Apply the update rule
            self.current_state = self.update_rule(self.current_state)
            
            # Store the new state and count of live cells
            self.state_history.append(self.current_state.copy())
            live_cells.append(np.sum(self.current_state))
        
        return live_cells
    
    def get_state_at_day(self, day: int) -> np.ndarray:
        """Get the grid state at a specific day.
        
        Args:
            day: The day (generation) to retrieve (0-indexed).
            
        Returns:
            The grid state at the specified day.
        """
        if not self.state_history or day >= len(self.state_history):
            raise ValueError(f"No data for day {day}. Run the simulation first.")
        
        return self.state_history[day]
    
    def get_all_states(self) -> List[np.ndarray]:
        """Get the complete history of grid states.
        
        Returns:
            A list of grid states for each generation.
        """
        if not self.state_history:
            raise ValueError("No simulation results available. Run the simulation first.")
        
        return self.state_history
    
    def detect_stable_pattern(self, max_cycle_length: int = 10) -> Optional[int]:
        """Detect if the simulation has reached a stable pattern or cycle.
        
        Args:
            max_cycle_length: Maximum cycle length to check for.
            
        Returns:
            The cycle length if a cycle is detected, 0 for a stable state, or None if no pattern detected.
        """
        if len(self.state_history) < max_cycle_length + 1:
            return None
        
        # Check for stable state (no change)
        latest_state = self.state_history[-1]
        if np.array_equal(latest_state, self.state_history[-2]):
            return 0
        
        # Check for cycles
        for cycle_len in range(1, max_cycle_length + 1):
            is_cycle = True
            for offset in range(1, cycle_len + 1):
                if not np.array_equal(self.state_history[-offset], self.state_history[-offset-cycle_len]):
                    is_cycle = False
                    break
            if is_cycle:
                return cycle_len
        
        return None
    
    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        super().reset()
        self.current_state = self.initial_state.copy()
        self.state_history = []
    
    @classmethod
    def get_parameters_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about the parameters required by this simulation.
        
        Returns:
            A dictionary mapping parameter names to their metadata.
        """
        # Get base parameters from parent class
        params = super().get_parameters_info()
        
        # Add class-specific parameters
        params.update({
            'grid_size': {
                'type': 'Tuple[int, int]',
                'description': 'The dimensions of the grid as (rows, columns)',
                'required': False,
                'default': (50, 50)
            },
            'initial_state': {
                'type': 'np.ndarray',
                'description': 'Initial configuration of the grid. If None, a random grid will be generated',
                'required': False,
                'default': None
            },
            'initial_density': {
                'type': 'float',
                'description': 'Probability of a cell being alive in the random initial state',
                'required': False,
                'default': 0.3
            },
            'rule': {
                'type': 'Union[str, Callable]',
                'description': 'Either a string for a predefined ruleset or a custom function',
                'required': False,
                'default': 'game_of_life'
            },
            'boundary': {
                'type': 'str',
                'description': 'Boundary condition (\'periodic\' or \'fixed\')',
                'required': False,
                'default': 'periodic'
            }
        })
        
        return params