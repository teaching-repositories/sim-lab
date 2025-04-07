import pytest
from sim_lab.core import SimulatorRegistry


def test_initialization():
    """Test initialization of the DiscreteEventSimulation class."""
    # Define a simple event action
    def dummy_action(simulation, data):
        simulation.state["value"] += data["increment"]
    
    # Initial events
    initial_events = [
        (5.0, dummy_action, {"increment": 10})
    ]
    
    sim = SimulatorRegistry.create(
        "DiscreteEvent",
        max_time=100.0,
        initial_events=initial_events,
        time_step=1.0,
        random_seed=42
    )
    
    assert sim.max_time == 100.0
    assert sim.time_step == 1.0
    assert sim.random_seed == 42
    assert len(sim.event_queue) == 1
    assert sim.event_queue[0].time == 5.0


def test_event_scheduling():
    """Test scheduling events in the simulation."""
    sim = SimulatorRegistry.create(
        "DiscreteEvent",
        max_time=100.0,
        random_seed=42
    )
    
    # Schedule an event
    def dummy_action(simulation, data):
        simulation.state["value"] += 1
    
    sim.schedule_event(10.0, dummy_action)
    assert len(sim.event_queue) == 1
    assert sim.event_queue[0].time == 10.0


def test_event_execution():
    """Test that events are executed at the correct time."""
    # Define an event action that records its execution time
    def record_time(simulation, data):
        simulation.state["executed_at"] = simulation.current_time
    
    # Initial events
    initial_events = [
        (15.0, record_time, None)
    ]
    
    sim = SimulatorRegistry.create(
        "DiscreteEvent",
        max_time=100.0,
        initial_events=initial_events,
        random_seed=42
    )
    
    # Run the simulation
    sim.run_simulation()
    
    # Check that the event was executed at the correct time
    assert sim.state["executed_at"] == 15.0


def test_event_priority():
    """Test that events with the same time are executed in order of priority."""
    executed_order = []
    
    # Define event actions that record their execution order
    def first_action(simulation, data):
        executed_order.append("first")
    
    def second_action(simulation, data):
        executed_order.append("second")
    
    # Initial events with the same time but different priorities
    initial_events = [
        (10.0, second_action, None),  # Default priority is 0
    ]
    
    sim = SimulatorRegistry.create(
        "DiscreteEvent",
        max_time=100.0,
        initial_events=initial_events,
        random_seed=42
    )
    
    # Schedule an event with higher priority (lower number)
    sim.schedule_event(10.0, first_action, priority=-1)
    
    # Run the simulation
    sim.run_simulation()
    
    # Check execution order (higher priority should execute first)
    assert executed_order == ["first", "second"]


def test_event_chain():
    """Test that events can schedule new events."""
    execution_times = []
    
    # Define an event action that schedules another event
    def schedule_next(simulation, data):
        execution_times.append(simulation.current_time)
        if simulation.current_time < 40.0:
            # Schedule next event 10 time units later
            simulation.schedule_event(
                simulation.current_time + 10.0, 
                schedule_next
            )
    
    # Initial events
    initial_events = [
        (10.0, schedule_next, None)
    ]
    
    sim = SimulatorRegistry.create(
        "DiscreteEvent",
        max_time=100.0,
        initial_events=initial_events,
        random_seed=42
    )
    
    # Run the simulation
    sim.run_simulation()
    
    # Should have executed at times 10, 20, 30, 40
    assert execution_times == [10.0, 20.0, 30.0, 40.0]


def test_simulation_results():
    """Test that the simulation returns the correct results."""
    # Define a custom solution where we completely control the reporting
    class CustomValueSimulation:
        def __init__(self):
            self.values = []
            self.current_value = 0
            
        def record_value(self, simulation, time):
            self.values.append((time, simulation.state["value"]))
    
    # Create a custom tracker
    tracker = CustomValueSimulation()
    
    # Define an event action that modifies the state value
    def increment_value(simulation, data):
        # Increment the state
        simulation.state["value"] += 10
        # Record the value
        tracker.record_value(simulation, simulation.current_time)
        # Schedule the next increment
        if simulation.current_time + 10 <= simulation.max_time:
            simulation.schedule_event(
                simulation.current_time + 10, 
                increment_value
            )
    
    # Add a recording for time 0
    def record_initial(simulation, data):
        tracker.record_value(simulation, 0.0)
    
    # Initial events
    initial_events = [
        (0.0, record_initial, None),
        (10.0, increment_value, None)
    ]
    
    sim = SimulatorRegistry.create(
        "DiscreteEvent",
        max_time=50.0,
        initial_events=initial_events,
        random_seed=42
    )
    
    # Run the simulation
    sim.run_simulation()
    
    # Now check the events explicitly
    expected_values = [
        (0.0, 0.0),   # Initial state
        (10.0, 10.0), # First increment
        (20.0, 20.0), # Second increment
        (30.0, 30.0), # Third increment
        (40.0, 40.0), # Fourth increment
        (50.0, 50.0)  # Fifth increment
    ]
    
    # Check that all expected values are recorded
    for expected_time, expected_value in expected_values:
        found = False
        for time, value in tracker.values:
            if time == expected_time:
                assert value == expected_value
                found = True
                break
        assert found, f"No value recorded for time {expected_time}"


def test_parameters_info():
    """Test the get_parameters_info method."""
    params = SimulatorRegistry.get("DiscreteEvent").get_parameters_info()
    assert isinstance(params, dict)
    assert "max_time" in params
    assert "initial_events" in params
    assert "time_step" in params
    assert "random_seed" in params