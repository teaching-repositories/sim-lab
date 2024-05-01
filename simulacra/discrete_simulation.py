from .base_simulation import BaseSimulation
from typing import Optional


class DiscreteEventSimulation(BaseSimulation):
    """
    Represents a discrete event simulation where updates occur at discrete time steps or based on specific events.
    """

    def __init__(self, time_step: float, units: str, event_description: str, random_seed: Optional[int] = None) -> None:
        """
        Initialises a discrete event simulation with specific attributes.

        Parameters:
            time_step (float): The time interval at which events are processed.
            units (str): The units of measurement for output data.
            event_description (str): Descriptive details about the events being simulated.
            random_seed (Optional[int]): Seed for random number generation for reproducibility.
        """
        super().__init__(time_step, units, random_seed)
        self.event_description = event_description

    def simulate_timestep(self) -> None:
        """
        Defines how the simulation updates at each discrete time step or event.
        """
        pass  # Placeholder for implementation in subclasses

    def run_simulation(self) -> None:
        """
        Runs the simulation by iterating over defined time steps or events.
        """
        for _ in range(self.days):
            self.simulate_timestep()
