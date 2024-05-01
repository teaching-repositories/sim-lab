import random
from typing import Optional

class BaseSimulation:
    """
    Base class for creating different types of simulations.

    Attributes:
        time_step (float): The time step between simulation points.
        units (str): The measurement units for the simulation results.
        random_seed (Optional[int]): An optional random seed for reproducibility.

    Methods:
        run_simulation: Abstract method to run the simulation.
    """

    def __init__(self, time_step: float, units: str, random_seed: Optional[int] = None) -> None:
        """
        Initialises the BaseSimulation with given time step, units, and optional random seed.
        """
        self.time_step = time_step
        self.units = units
        self.random_seed = random_seed
        if random_seed:
            random.seed(random_seed)  # For reproducibility 

    def run_simulation(self) -> None:
        """
        Abstract method that should be implemented by subclasses to run the simulation.
        """
        raise NotImplementedError("run_simulation must be implemented by the subclass")