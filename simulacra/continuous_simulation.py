from .base_simulation import BaseSimulation
from typing import Optional

class ContinuousSimulation(BaseSimulation):
    """
    Represents a simulation that updates continuously over time.
    """

    def __init__(self, time_step: float, units: str, simulation_detail: str, random_seed: Optional[int] = None) -> None:
        """
        Initialises a continuous simulation with specific attributes.

        Parameters:
            time_step (float): The time step between updates in the simulation.
            units (str): The units of measurement for output data.
            simulation_detail (str): Details specific to continuous simulations.
            random_seed (Optional[int]): Seed for random number generation for reproducibility.
        """
        super().__init__(time_step, units, random_seed)
        self.simulation_detail = simulation_detail

    def run_simulation(self) -> None:
        """
        Runs the continuous simulation.
        """
        pass  # This method should be implemented by subclasses