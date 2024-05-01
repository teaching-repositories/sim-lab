from .continuous_simulation import ContinuousSimulation


class DiseaseSimulation(ContinuousSimulation):
    """
    Simulates the spread of a disease through a population over time, incorporating factors like infection rate and recovery rate.
    """

    def __init__(self, start_population: int, infection_rate: float, recovery_rate: float, days: int) -> None:
        """
        Initializes a new disease simulation.

        Parameters:
            start_population (int): The initial population at risk of infection.
            infection_rate (float): The rate at which the disease spreads among the population.
            recovery_rate (float): The rate at which infected individuals recover from the disease.
            days (int): The total number of days to simulate the disease spread.
        """
        super().__init__(days)
