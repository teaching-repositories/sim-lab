import numpy as np
from typing import List, Optional


class StockMarketSimulation:
    """
    A simulation class to model the fluctuations of stock prices over time, accounting for volatility,
    general market trends (drift), and specific market events.

    Attributes:
        start_price (float): The initial price of the stock.
        days (int): The duration of the simulation in days.
        volatility (float): The volatility of stock price changes, representing day-to-day variability.
        drift (float): The average daily price change, indicating the trend over time.
        event_day (Optional[int]): The specific day a major market event occurs (default is None).
        event_impact (float): The magnitude of the event's impact on stock prices (default is 0).
        random_seed (Optional[int]): The seed for the random number generator to ensure reproducibility (default is None).

    Methods:
        run_simulation(): Runs the simulation and returns a list of stock prices over the simulation period.
    """

    def __init__(
        self, start_price: float, days: int, volatility: float, drift: float,
        event_day: Optional[int] = None, event_impact: float = 0,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initializes the StockMarketSimulation with all necessary parameters.

        Parameters:
            start_price (float): The initial stock price.
            days (int): The total number of days to simulate.
            volatility (float): The volatility of the stock price, representing the randomness of day-to-day price changes.
            drift (float): The expected daily percentage change in price, which can be positive or negative.
            event_day (Optional[int]): Day on which a major market event occurs (defaults to None).
            event_impact (float): The severity of the market event, affecting prices multiplicatively.
            random_seed (Optional[int]): Seed for the random number generator to ensure reproducible results (defaults to None).
        """
        self.start_price = start_price
        self.days = days
        self.volatility = volatility
        self.drift = drift
        self.event_day = event_day
        self.event_impact = event_impact
        self.random_seed = random_seed

    def run_simulation(self) -> List[float]:
        """
        Simulates the stock price over a specified number of days based on the initial settings.

        Returns:
            List[float]: A list containing the stock prices for each day of the simulation.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        prices = [self.start_price]
        for day in range(1, self.days):
            previous_price = prices[-1]
            random_change = np.random.normal(self.drift, self.volatility)
            new_price = previous_price * (1 + random_change)

            if day == self.event_day:
                new_price = previous_price * (1 + self.event_impact)

            prices.append(new_price)

        return prices
