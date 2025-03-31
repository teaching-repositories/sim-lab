"""Base class for all simulations."""

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Union


class BaseSimulation(ABC):
    """Base class for all SimNexus simulations.
    
    This abstract class defines the common interface and utility methods
    for all simulation types in the SimNexus package.
    
    Attributes:
        random_seed (Optional[int]): Seed for random number generation to ensure reproducible results
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the base simulation.
        
        Args:
            random_seed: Seed for random number generation. If None, random results will vary.
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
    
    @abstractmethod
    def run_simulation(self) -> List[Union[float, int]]:
        """Run the simulation and return results.
        
        This method must be implemented by all simulation subclasses.
        
        Returns:
            A list of values representing the simulation results over time.
        """
        pass
    
    def reset(self) -> None:
        """Reset the simulation to its initial state.
        
        This allows a simulation instance to be re-run with the same parameters.
        """
        if self.random_seed is not None:
            random.seed(self.random_seed)