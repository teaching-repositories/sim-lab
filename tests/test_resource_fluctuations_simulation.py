import pytest
from sim_lab.core import SimulatorRegistry


def test_initialization():
    """Test initialization of the ResourceFluctuationsSimulation class."""
    sim = SimulatorRegistry.create(
        "ResourceFluctuations",
        start_price=100, 
        days=365, 
        volatility=0.01, 
        drift=0.0001,
        supply_disruption_day=180, 
        disruption_severity=0.2, 
        random_seed=42
    )
    assert sim.start_price == 100
    assert sim.days == 365
    assert sim.volatility == 0.01
    assert sim.drift == 0.0001
    assert sim.supply_disruption_day == 180
    assert sim.disruption_severity == 0.2
    assert sim.random_seed == 42


def test_run_simulation_output_length():
    """Test that the simulation returns the correct number of price points."""
    sim = SimulatorRegistry.create(
        "ResourceFluctuations",
        start_price=100, 
        days=365, 
        volatility=0.01, 
        drift=0.0001, 
        random_seed=42
    )
    prices = sim.run_simulation()
    assert len(prices) == 365


def test_run_simulation_reproducibility():
    """Test that the simulation results are reproducible with the same random seed."""
    sim1 = SimulatorRegistry.create(
        "ResourceFluctuations",
        start_price=100, 
        days=365, 
        volatility=0.01, 
        drift=0.0001, 
        random_seed=42
    )
    sim2 = SimulatorRegistry.create(
        "ResourceFluctuations",
        start_price=100, 
        days=365, 
        volatility=0.01, 
        drift=0.0001, 
        random_seed=42
    )
    prices1 = sim1.run_simulation()
    prices2 = sim2.run_simulation()
    assert prices1 == prices2


def test_supply_disruption_effect():
    """Test the effect of a supply disruption on the specified day."""
    sim = SimulatorRegistry.create(
        "ResourceFluctuations",
        start_price=100, 
        days=365, 
        volatility=0.01, 
        drift=0.0001,
        supply_disruption_day=180, 
        disruption_severity=0.1, 
        random_seed=42
    )
    prices = sim.run_simulation()
    
    # Check that the price on the disruption day is impacted as expected
    no_disruption_price = prices[179]  # The price the day before the disruption
    disruption_price = prices[180]     # The price on the disruption day
    expected_disruption_price = no_disruption_price * (1 + sim.disruption_severity)
    assert disruption_price == pytest.approx(expected_disruption_price)


def test_parameters_info():
    """Test the get_parameters_info method."""
    params = SimulatorRegistry.get("ResourceFluctuations").get_parameters_info()
    assert isinstance(params, dict)
    assert "days" in params
    assert "start_price" in params
    assert "volatility" in params
    assert "drift" in params
    assert "supply_disruption_day" in params
    assert "disruption_severity" in params
    assert "random_seed" in params