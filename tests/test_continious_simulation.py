import pytest
import numpy as np
from simulacra.continuous_simulation import ContinuousSimulation

@pytest.fixture
def simulation_instance():
    # Create a ContinuousSimulation instance with known parameters for testing
    return ContinuousSimulation(start_value=100, timesteps=10, volatility=0.1, drift=0.05)

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

    # Test if the first element of the generated array matches the start value
    assert simulation_result[0] == simulation_instance.start_value

    # Test if all elements of the generated array are of type float
    assert all(isinstance(value, float) for value in simulation_result)

    # You can add more specific tests depending on your requirements
