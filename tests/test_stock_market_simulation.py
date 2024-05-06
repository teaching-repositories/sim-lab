import pytest
from simulacra import StockMarketSimulation


def test_initialization():
    """Test initialization of the StockMarketSimulation class."""
    sim = StockMarketSimulation(
        start_price=100, days=365, volatility=0.03, drift=-0.001,
        event_day=100, event_impact=-0.2, random_seed=42
    )
    assert sim.start_price == 100
    assert sim.days == 365
    assert sim.volatility == 0.03
    assert sim.drift == -0.001
    assert sim.event_day == 100
    assert sim.event_impact == -0.2
    assert sim.random_seed == 42

def test_run_simulation_output_length():
    """Test that the simulation returns the correct number of price points."""
    sim = StockMarketSimulation(
        start_price=100, days=365, volatility=0.03, drift=-0.001, random_seed=42
    )
    prices = sim.run_simulation()
    assert len(prices) == 365

def test_run_simulation_reproducibility():
    """Test that the simulation results are reproducible with the same random seed."""
    sim1 = StockMarketSimulation(
        start_price=100, days=365, volatility=0.03, drift=-0.001, random_seed=42
    )
    sim2 = StockMarketSimulation(
        start_price=100, days=365, volatility=0.03, drift=-0.001, random_seed=42
    )
    prices1 = sim1.run_simulation()
    prices2 = sim2.run_simulation()
    assert prices1 == prices2

def test_event_impact():
    """Test the effect of a market event on the specified day."""
    sim = StockMarketSimulation(
        start_price=100, days=365, volatility=0.03, drift=-0.001,
        event_day=180, event_impact=-0.1, random_seed=42
    )
    prices = sim.run_simulation()
    # Assuming a reproducible result with a set random seed:
    # Check that the price on the event day is impacted as expected.
    expected_impact = prices[179] * (1 - 0.1)  # Expecting a 10% drop
    assert prices[180] == pytest.approx(expected_impact)

# Additional tests could include:
# - Testing the output type (ensure it's all floats or ints, as expected)
# - Testing edge cases like zero or negative values for parameters
# - Testing the handling of different types of input errors
