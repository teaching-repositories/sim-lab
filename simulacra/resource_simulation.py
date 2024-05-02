import numpy as np
from typing import List, Optional


class ResourceSimulation:
    """
    A simulation class to model the price fluctuations of a critical resource over time,
    considering factors like volatility, drift, and supply disruptions.

    Attributes:
        start_price (float): The initial price of the resource.
        days (int): The duration of the simulation in days.
        volatility (float): The volatility of price changes, representing day-to-day variability.
        drift (float): The average daily price change, indicating the trend over time.
        supply_disruption_day (Optional[int]): The specific day a supply disruption occurs (default is None).
        disruption_severity (float): The magnitude of the disruption's impact on price (default is 0).
        random_seed (Optional[int]): The seed for the random number generator to ensure reproducibility (default is None).

    Methods:
        run_simulation(): Runs the simulation and returns a list of prices over the simulation period.
    """

    def __init__(self, start_price: float, days: int, volatility: float, drift: float,
                 supply_disruption_day: Optional[int] = None, disruption_severity: float = 0,
                 random_seed: Optional[int] = None) -> None:
        """
        Initializes the ResourceSimulation with all necessary parameters.

        Parameters:
            start_price (float): The initial price of the resource.
            days (int): The total number of days to simulate.
            volatility (float): The volatility of the resource price, representing the randomness of day-to-day price changes.
            drift (float): The expected daily percentage change in price, which can be positive or negative.
            supply_disruption_day (Optional[int]): Day on which a supply disruption occurs (defaults to None).
            disruption_severity (float): The severity of the supply disruption, affecting prices multiplicatively.
            random_seed (Optional[int]): Seed for the random number generator to ensure reproducible results (defaults to None).
        """
        self.start_price = start_price
        self.days = days
        self.volatility = volatility
        self.drift = drift
        self.supply_disruption_day = supply_disruption_day
        self.disruption_severity = disruption_severity
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def run_simulation(self) -> List[float]:
        """
        Simulates the price of the resource over a specified number of days based on the initial settings.

        Returns:
            List[float]: A list containing the price of the resource for each day of the simulation.
        """
        prices = [self.start_price]
        for day in range(1, self.days):
            previous_price = prices[-1]
            random_change = np.random.normal(self.drift, self.volatility)
            new_price = previous_price * (1 + random_change)

            if day == self.supply_disruption_day:
                new_price *= (1 + self.disruption_severity)

            prices.append(new_price)

        return prices
