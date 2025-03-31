from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
import numpy as np


class BaseSimulation(ABC):
    """
    Abstract base class for all simulation types within SimNexus.
    
    This class defines the common interface for all simulations and
    provides shared functionality for initialization, randomization,
    and results handling.
    
    Attributes:
        days (int): The duration of the simulation in days.
        random_seed (Optional[int]): The seed for reproducible simulations.
    """
    
    def __init__(self, days: int, random_seed: Optional[int] = None) -> None:
        """
        Initialize the base simulation.
        
        Args:
            days (int): The duration of the simulation in days.
            random_seed (Optional[int]): The seed for reproducible simulations.
        """
        self.days = days
        self.random_seed = random_seed
        self._results: Dict[str, Any] = {}
        
    def _set_random_seed(self) -> None:
        """Set the random seed if one was provided."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
    @abstractmethod
    def run_simulation(self) -> Any:
        """
        Run the simulation and return the results.
        
        This method must be implemented by subclasses.
        
        Returns:
            Any: The simulation results.
        """
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the simulation results.
        
        Returns:
            Dict[str, Any]: The simulation results.
        """
        return self._results
    
    def export_results(self, file_path: str, format: str = 'csv') -> None:
        """
        Export the simulation results to a file.
        
        Args:
            file_path (str): The path to save the results to.
            format (str): The format to use ('csv', 'json', etc.)
        """
        # Placeholder for future implementation
        pass