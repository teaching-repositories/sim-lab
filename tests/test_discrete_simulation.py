import pytest
import numpy as np
from simulacra.discrete_simulation import DiscreteSimulation

@pytest.fixture
def simulation_instance():
    # Create a DiscreteSimulation instance with known parameters for testing
    return DiscreteSimulation(start_value=100, timesteps=10, possible_values=[95, 100, 105], probabilities=[0.2, 0.6, 0.2])

def test_generate_values(simulation_instance):
    # Test if generate_values method returns a list of correct length
    values = simulation_instance.generate_values()
    assert len(values) == simulation_instance.timesteps

def test_run_simulation(simulation_instance):
    # Test if run_simulation method returns a numpy array
    simulation_result = simulation_instance.run_simulation()
    assert isinstance(simulation_result, np.ndarray)

    # Test if the length of the generated array matches the number of timesteps
    assert len(simulation_result) == simulation_instance.timesteps

    # Test if all elements of the generated array are one of the possible values
    assert all(value in simulation_instance.possible_values for value in simulation_result)

    # You can add more specific tests depending on your requirements
