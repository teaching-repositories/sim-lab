from .continuous_simulation import ContinuousSimulation
from typing import Optional


class ProductPopularitySimulation(ContinuousSimulation):
    """
    Simulates the rise and fall of product popularity over time based on initial popularity and growth factors.
    """

    def __init__(self, initial_popularity: int, days: int, growth_rate: float, event_day: Optional[int] = None, event_impact: Optional[float] = None) -> None:
        """
        Initialises a new product popularity simulation.

        Parameters:
            initial_popularity (int): The initial popularity level of the product.
            days (int): The total number of days to simulate the product's popularity.
            growth_rate (float): The rate at which the product's popularity grows or declines.
            event_day (Optional[int]): A specific day an event occurs that might affect the product's popularity.
            event_impact (Optional[float]): The impact of the event on the product's popularity.
        """
        super().__init__(days)