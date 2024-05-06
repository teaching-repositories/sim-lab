import numpy as np
from typing import List, Optional


class ProductPopularitySimulation:
    """
    A simulation class to model the dynamics of product popularity over time,
    incorporating factors like natural growth, marketing impact, and promotional campaigns.

    Attributes:
        start_demand (int): Initial demand for the product.
        days (int): Duration of the simulation in days.
        growth_rate (float): Natural growth rate of product demand.
        marketing_impact (float): Impact of ongoing marketing efforts on demand.
        promotion_day (Optional[int]): Day on which a major marketing campaign starts (default is None).
        promotion_effectiveness (float): Effectiveness of the marketing campaign.
        random_seed (Optional[int]): The seed for the random number generator to ensure reproducibility (default is None).

    Methods:
        run_simulation(): Runs the simulation and returns a list of demand values over time.
    """

    def __init__(
        self, start_demand: float, days: int, growth_rate: float, marketing_impact: float,
        promotion_day: Optional[int] = None, promotion_effectiveness: float = 0,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initializes the ProductPopularitySimulation with all necessary parameters.

        Parameters:
            start_demand (int): The initial level of demand for the product.
            days (int): The total number of days to simulate.
            growth_rate (float): The natural daily growth rate of demand, as a decimal.
            marketing_impact (float): Daily impact of marketing on demand, as a decimal.
            promotion_day (Optional[int]): The specific day on which a promotional event occurs (defaults to None).
            promotion_effectiveness (float): Multiplicative impact of the promotion on demand.
            random_seed (Optional[int]): Seed for the random number generator to ensure reproducible results (defaults to None).
        """
        self.start_demand = start_demand
        self.days = days
        self.growth_rate = growth_rate
        self.marketing_impact = marketing_impact
        self.promotion_day = promotion_day
        self.promotion_effectiveness = promotion_effectiveness
        self.random_seed = random_seed

    def run_simulation(self) -> List[float]:
        """
        Simulates the demand for a product over a specified number of days based on the initial settings.

        Returns:
            List[int]: A list containing the demand for the product for each day of the simulation.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        demand = [self.start_demand]
        for day in range(1, self.days):
            previous_demand = demand[-1]
            natural_growth = previous_demand * (1 + self.growth_rate)
            marketing_influence = previous_demand * self.marketing_impact

            new_demand = natural_growth + marketing_influence

            if day == self.promotion_day:
                new_demand *= (1 + self.promotion_effectiveness)

            demand.append(new_demand)

        return demand
