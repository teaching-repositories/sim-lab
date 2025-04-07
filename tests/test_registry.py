import pytest
from sim_lab.core import SimulatorRegistry, BaseSimulation


def test_registry_initialization():
    """Test that the registry is initialized properly."""
    # Registry should exist and be a dict
    assert isinstance(SimulatorRegistry._registry, dict)
    # Should have some simulators registered
    assert len(SimulatorRegistry._registry) > 0


def test_list_simulators():
    """Test that list_simulators returns all registered simulators."""
    simulators = SimulatorRegistry.list_simulators()
    assert isinstance(simulators, list)
    assert len(simulators) > 0
    assert "StockMarket" in simulators
    assert "ResourceFluctuations" in simulators
    assert "ProductPopularity" in simulators


def test_get_simulator():
    """Test that we can get a simulator class by name."""
    # Get a simulator class
    sim_class = SimulatorRegistry.get("StockMarket")
    # Should be a class and a subclass of BaseSimulation
    assert isinstance(sim_class, type)
    assert issubclass(sim_class, BaseSimulation)


def test_create_simulator():
    """Test that we can create a simulator instance."""
    # Create a simulator
    sim = SimulatorRegistry.create(
        "StockMarket",
        start_price=100,
        days=365,
        volatility=0.02,
        drift=0.001,
        random_seed=42
    )
    # Should be an instance of BaseSimulation
    assert isinstance(sim, BaseSimulation)
    # Should have the correct attributes
    assert sim.days == 365
    assert sim.random_seed == 42


def test_register_simulator():
    """Test registering a new simulator class."""
    # Define a simple test simulator
    @SimulatorRegistry.register("TestSim")
    class TestSimulation(BaseSimulation):
        def __init__(self, days, test_param=None, random_seed=None):
            super().__init__(days=days, random_seed=random_seed)
            self.test_param = test_param

        def run_simulation(self):
            return [0] * self.days

    # Should be in the registry
    assert "TestSim" in SimulatorRegistry.list_simulators()
    
    # Create an instance
    sim = SimulatorRegistry.create("TestSim", days=10, test_param="test", random_seed=42)
    assert isinstance(sim, TestSimulation)
    assert sim.test_param == "test"
    assert sim.days == 10


def test_unregister_simulator():
    """Test unregistering a simulator class."""
    # First register a temporary test simulator
    @SimulatorRegistry.register("TempSim")
    class TempSimulation(BaseSimulation):
        def run_simulation(self):
            return [0] * self.days

    # Should be in the registry
    assert "TempSim" in SimulatorRegistry.list_simulators()
    
    # Unregister it
    SimulatorRegistry.unregister("TempSim")
    
    # Should no longer be in the registry
    assert "TempSim" not in SimulatorRegistry.list_simulators()
    
    # Trying to get it should raise a KeyError
    with pytest.raises(KeyError):
        SimulatorRegistry.get("TempSim")


def test_get_nonexistent_simulator():
    """Test that getting a non-existent simulator raises a KeyError."""
    with pytest.raises(KeyError):
        SimulatorRegistry.get("NonExistentSimulator")


def test_create_nonexistent_simulator():
    """Test that creating a non-existent simulator raises a KeyError."""
    with pytest.raises(KeyError):
        SimulatorRegistry.create("NonExistentSimulator", days=10)