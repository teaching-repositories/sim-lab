import pytest
from simulacra import ProductPopularitySimulation


def test_initialization():
    """Test initialization of the ProductPopularitySimulation class."""
    sim = ProductPopularitySimulation(
        start_demand=500, days=180, growth_rate=0.02, marketing_impact=0.1,
        promotion_day=30, promotion_effectiveness=0.5, random_seed=42
    )
    assert sim.start_demand == 500
    assert sim.days == 180
    assert sim.growth_rate == 0.02
    assert sim.marketing_impact == 0.1
    assert sim.promotion_day == 30
    assert sim.promotion_effectiveness == 0.5
    assert sim.random_seed == 42

def test_run_simulation_output_length():
    """Test that the simulation returns the correct number of demand points."""
    sim = ProductPopularitySimulation(
        start_demand=500, days=180, growth_rate=0.02, marketing_impact=0.1, random_seed=42
    )
    demand = sim.run_simulation()
    assert len(demand) == 180

def test_run_simulation_reproducibility():
    """Test that the simulation results are reproducible with the same random seed."""
    sim1 = ProductPopularitySimulation(
        start_demand=500, days=180, growth_rate=0.02, marketing_impact=0.1, random_seed=42
    )
    sim2 = ProductPopularitySimulation(
        start_demand=500, days=180, growth_rate=0.02, marketing_impact=0.1, random_seed=42
    )
    demand1 = sim1.run_simulation()
    demand2 = sim2.run_simulation()
    assert demand1 == demand2

'''
def test_promotion_effectiveness():
    """Test the effect of a promotional campaign on the specified day."""
    sim = ProductPopularitySimulation(
        start_demand=500, days=180, growth_rate=0.02, marketing_impact=0.1,
        promotion_day=30, promotion_effectiveness=0.5, random_seed=42
    )
    demand = sim.run_simulation()
    # Assuming a reproducible result with a set random seed:
    # Check that the demand on the promotion day increases as expected.
    expected_increase = demand[29] * (1 + sim.promotion_effectiveness)  # Day 30 is index 29
    assert demand[30] == pytest.approx(expected_increase)
'''

# Additional tests could include:
# - Testing the output type (ensure it's all floats or ints, as expected)
# - Testing edge cases like zero or negative values for parameters
# - Testing the handling of different types of input errors
