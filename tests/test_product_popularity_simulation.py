import pytest
from sim_lab.core import SimulatorRegistry


def test_initialization():
    """Test initialization of the ProductPopularitySimulation class."""
    sim = SimulatorRegistry.create(
        "ProductPopularity",
        start_demand=500, 
        days=180, 
        growth_rate=0.02, 
        marketing_impact=0.1,
        promotion_day=30, 
        promotion_effectiveness=0.5, 
        random_seed=42
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
    sim = SimulatorRegistry.create(
        "ProductPopularity",
        start_demand=500, 
        days=180, 
        growth_rate=0.02, 
        marketing_impact=0.1, 
        random_seed=42
    )
    demand = sim.run_simulation()
    assert len(demand) == 180


def test_run_simulation_reproducibility():
    """Test that the simulation results are reproducible with the same random seed."""
    sim1 = SimulatorRegistry.create(
        "ProductPopularity",
        start_demand=500, 
        days=180, 
        growth_rate=0.02, 
        marketing_impact=0.1, 
        random_seed=42
    )
    sim2 = SimulatorRegistry.create(
        "ProductPopularity",
        start_demand=500, 
        days=180, 
        growth_rate=0.02, 
        marketing_impact=0.1, 
        random_seed=42
    )
    demand1 = sim1.run_simulation()
    demand2 = sim2.run_simulation()
    assert demand1 == demand2


def test_promotion_effectiveness():
    """Test the effect of a promotional campaign on the specified day."""
    sim = SimulatorRegistry.create(
        "ProductPopularity",
        start_demand=500, 
        days=180, 
        growth_rate=0.02, 
        marketing_impact=0.1,
        promotion_day=30, 
        promotion_effectiveness=0.5, 
        random_seed=42
    )
    demand = sim.run_simulation()
    
    # Check day before promotion (index 29)
    day_before_promotion = demand[29]
    # Calculate the expected demand for promotion day (index 30)
    natural_growth = day_before_promotion * (1 + sim.growth_rate)
    marketing_influence = day_before_promotion * sim.marketing_impact
    expected_demand = (natural_growth + marketing_influence) * (1 + sim.promotion_effectiveness)
    
    assert demand[30] == pytest.approx(expected_demand)


def test_growth_over_time():
    """Test that demand grows over time with positive growth rate."""
    sim = SimulatorRegistry.create(
        "ProductPopularity",
        start_demand=100, 
        days=100, 
        growth_rate=0.01, 
        marketing_impact=0.0,  # No marketing to isolate growth
        random_seed=42
    )
    demand = sim.run_simulation()
    
    # Should have positive growth rate
    assert demand[-1] > demand[0]
    
    # Check a specific day's calculation
    day_10_expected = 100 * (1.01 ** 10)  # 1% growth compounded 10 times
    # Allow some floating-point tolerance
    assert demand[10] == pytest.approx(day_10_expected, rel=1e-2)


def test_parameters_info():
    """Test the get_parameters_info method."""
    params = SimulatorRegistry.get("ProductPopularity").get_parameters_info()
    assert isinstance(params, dict)
    assert "days" in params
    assert "start_demand" in params
    assert "growth_rate" in params
    assert "marketing_impact" in params
    assert "promotion_day" in params
    assert "promotion_effectiveness" in params
    assert "random_seed" in params