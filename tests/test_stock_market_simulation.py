import pytest
from sim_lab.core import SimulatorRegistry


def test_initialization():
    """Test initialization of the StockMarketSimulation class."""
    sim = SimulatorRegistry.create(
        "StockMarket",
        start_price=100, 
        days=365, 
        volatility=0.03, 
        drift=-0.001,
        event_day=100, 
        event_impact=-0.2, 
        random_seed=42
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
    sim = SimulatorRegistry.create(
        "StockMarket",
        start_price=100, 
        days=365, 
        volatility=0.03, 
        drift=-0.001, 
        random_seed=42
    )
    prices = sim.run_simulation()
    assert len(prices) == 365


def test_run_simulation_reproducibility():
    """Test that the simulation results are reproducible with the same random seed."""
    sim1 = SimulatorRegistry.create(
        "StockMarket",
        start_price=100, 
        days=365, 
        volatility=0.03, 
        drift=-0.001, 
        random_seed=42
    )
    sim2 = SimulatorRegistry.create(
        "StockMarket",
        start_price=100, 
        days=365, 
        volatility=0.03, 
        drift=-0.001, 
        random_seed=42
    )
    prices1 = sim1.run_simulation()
    prices2 = sim2.run_simulation()
    assert prices1 == prices2


def test_event_impact():
    """Test the effect of a market event on the specified day."""
    sim = SimulatorRegistry.create(
        "StockMarket",
        start_price=100, 
        days=365, 
        volatility=0.03, 
        drift=-0.001,
        event_day=180, 
        event_impact=-0.1, 
        random_seed=42
    )
    prices = sim.run_simulation()
    # Check that the price on the event day is impacted as expected
    no_event_price = prices[179]  # The price the day before the event
    event_price = prices[180]     # The price on the event day
    expected_event_price = no_event_price * (1 + sim.event_impact)
    assert event_price == pytest.approx(expected_event_price)


def test_base_simulation_methods():
    """Test methods inherited from BaseSimulation."""
    sim = SimulatorRegistry.create(
        "StockMarket",
        start_price=100, 
        days=365, 
        volatility=0.03, 
        drift=-0.001,
        random_seed=42
    )
    
    # Test reset method
    sim.run_simulation()  # run once
    result1 = sim.run_simulation()  # run again without reset
    
    sim.reset()  # reset state
    result2 = sim.run_simulation()  # run after reset
    
    assert result1 == result2  # should get same results after reset
    
    # Test get_parameters_info
    params = sim.get_parameters_info()
    assert isinstance(params, dict)
    assert "days" in params
    assert "start_price" in params
    assert "volatility" in params
    assert "drift" in params
    assert "event_day" in params
    assert "event_impact" in params
    assert "random_seed" in params