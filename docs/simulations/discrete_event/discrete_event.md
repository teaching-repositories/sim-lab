# Discrete Event Simulation

Discrete Event Simulation (DES) is a modeling approach that represents systems as a sequence of events occurring at specific points in time, with each event changing the system state and potentially scheduling future events.

## Overview

The Discrete Event Simulation in SimLab provides a flexible framework for modeling event-driven systems with the following key features:

- Event scheduling with precise timing
- Priority-based event handling
- Custom event actions
- Efficient event processing using a priority queue
- Simulation state tracking over time
- Support for stochastic processes

## Basic Usage

```python
from sim_lab.core import SimulatorRegistry

# Define an event action function
def increment_value(simulation, data):
    # Increase the state value
    simulation.state["value"] += data["amount"]
    
    # Schedule the next event
    next_time = simulation.current_time + data["interval"]
    if next_time <= simulation.max_time:
        simulation.schedule_event(
            time=next_time,
            action=increment_value,
            data=data
        )

# Create initial events
initial_events = [
    (1.0, increment_value, {"amount": 5, "interval": 2.0})  # Start at t=1, repeat every 2 time units
]

# Create a discrete event simulation
sim = SimulatorRegistry.create(
    "DiscreteEvent",
    max_time=20.0,
    initial_events=initial_events,
    time_step=1.0,
    random_seed=42
)

# Run the simulation
results = sim.run_simulation()

# Print the results
for t, value in enumerate(results):
    print(f"Time {t}: {value}")
```

## Parameters

The Discrete Event Simulation accepts the following parameters:

| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `max_time` | float | The maximum simulation time | Yes | - |
| `initial_events` | List[Tuple] | Initial events as (time, action, data) tuples | No | None |
| `time_step` | float | Time step for recording results | No | 1.0 |
| `random_seed` | int | Seed for random number generation | No | None |

## Key Concepts

### Events

An event in the simulation is represented by the `Event` class, which includes:

- **Time**: When the event occurs
- **Action**: A function that executes when the event is processed
- **Priority**: For handling simultaneous events (lower number = higher priority)
- **Data**: Additional information associated with the event

### Event Actions

Event actions are functions with the signature:

```python
def event_action(simulation, data):
    # Modify simulation state
    simulation.state["key"] = new_value
    
    # Optionally schedule new events
    simulation.schedule_event(time, action, priority, data)
```

### Simulation State

The simulation maintains a state dictionary that can be modified by event actions:

```python
simulation.state = {"value": 0.0, "custom_key": "custom_value"}
```

### Event Queue

The event queue is a priority queue that schedules events in chronological order. When multiple events occur at the same time, they are processed in order of priority.

## Example: Bank Queue Simulation

```python
from sim_lab.core import SimulatorRegistry
import random
import matplotlib.pyplot as plt

# Customer arrival event
def customer_arrival(simulation, data):
    # Increment customer count
    simulation.state["total_customers"] += 1
    current_customer = simulation.state["total_customers"]
    
    # Add to the queue
    simulation.state["queue"].append(current_customer)
    queue_length = len(simulation.state["queue"])
    simulation.state["max_queue_length"] = max(queue_length, simulation.state["max_queue_length"])
    
    # Record queue length history
    simulation.state["queue_length_history"].append((simulation.current_time, queue_length))
    
    # If there are idle tellers, start service immediately
    if simulation.state["idle_tellers"] > 0:
        simulation.state["idle_tellers"] -= 1
        simulation.schedule_event(
            time=simulation.current_time,
            action=service_start,
            data=None
        )
    
    # Schedule the next arrival
    interarrival_time = random.expovariate(1.0 / data["mean_interarrival"])
    next_arrival_time = simulation.current_time + interarrival_time
    
    if next_arrival_time < simulation.max_time:
        simulation.schedule_event(
            time=next_arrival_time,
            action=customer_arrival,
            data=data
        )

# Service start event
def service_start(simulation, data):
    # Take the next customer from the queue
    if simulation.state["queue"]:
        customer = simulation.state["queue"].pop(0)
        
        # Record queue length after customer is removed
        queue_length = len(simulation.state["queue"])
        simulation.state["queue_length_history"].append((simulation.current_time, queue_length))
        
        # Generate service time and schedule service completion
        service_time = random.expovariate(1.0 / simulation.state["mean_service_time"])
        simulation.schedule_event(
            time=simulation.current_time + service_time,
            action=service_completion,
            data={"customer": customer, "service_time": service_time}
        )

# Service completion event
def service_completion(simulation, data):
    # Record service statistics
    wait_time = simulation.current_time - data["service_time"] - simulation.state["arrival_times"].get(data["customer"], 0)
    simulation.state["total_wait_time"] += wait_time
    simulation.state["completed_services"] += 1
    
    # Check if there are more customers in the queue
    if simulation.state["queue"]:
        # Serve the next customer
        simulation.schedule_event(
            time=simulation.current_time,
            action=service_start,
            data=None
        )
    else:
        # Teller becomes idle
        simulation.state["idle_tellers"] += 1

# Create bank simulation using the discrete event framework
def run_bank_simulation(mean_interarrival=5.0, mean_service_time=3.0, num_tellers=2, simulation_time=480):
    # Initial state
    initial_state = {
        "queue": [],                    # Customer queue
        "total_customers": 0,           # Total customers that arrived
        "completed_services": 0,        # Completed services
        "idle_tellers": num_tellers,    # Number of idle tellers
        "total_wait_time": 0,           # Total customer wait time
        "mean_service_time": mean_service_time,  # Mean service time
        "max_queue_length": 0,          # Maximum queue length
        "arrival_times": {},            # Customer arrival times
        "queue_length_history": [],     # Queue length over time
        "value": 0                      # Required for the base simulation
    }
    
    # Initial events
    initial_events = [
        (0.0, customer_arrival, {"mean_interarrival": mean_interarrival})
    ]
    
    # Create and run the simulation
    sim = SimulatorRegistry.create(
        "DiscreteEvent",
        max_time=simulation_time,
        initial_events=initial_events,
        time_step=1.0,
        random_seed=42
    )
    
    # Set the initial state
    sim.state = initial_state
    
    # Run the simulation
    sim.run_simulation()
    
    return sim

# Run the bank simulation
bank_sim = run_bank_simulation(
    mean_interarrival=5.0,   # Mean time between customer arrivals (minutes)
    mean_service_time=3.0,   # Mean time to serve a customer (minutes)
    num_tellers=2,           # Number of tellers
    simulation_time=480      # Simulation time (minutes) - 8 hour day
)

# Extract and analyze results
total_customers = bank_sim.state["total_customers"]
completed_services = bank_sim.state["completed_services"]
avg_wait_time = bank_sim.state["total_wait_time"] / completed_services if completed_services > 0 else 0
max_queue = bank_sim.state["max_queue_length"]

print(f"Total customers: {total_customers}")
print(f"Completed services: {completed_services}")
print(f"Average wait time: {avg_wait_time:.2f} minutes")
print(f"Maximum queue length: {max_queue}")

# Plot queue length over time
times, queue_lengths = zip(*bank_sim.state["queue_length_history"])
plt.figure(figsize=(12, 6))
plt.step(times, queue_lengths, where='post')
plt.title('Bank Queue Length Throughout the Day')
plt.xlabel('Time (minutes)')
plt.ylabel('Queue Length')
plt.grid(True)
plt.show()
```

## Example: Network Packet Simulation

```python
from sim_lab.core import SimulatorRegistry
import random
import matplotlib.pyplot as plt

# Packet arrival event
def packet_arrival(simulation, data):
    # Add packet to the buffer
    packet_size = random.expovariate(1.0 / data["mean_packet_size"])
    simulation.state["buffer"].append(packet_size)
    
    # Update buffer statistics
    simulation.state["total_packets"] += 1
    current_buffer_size = sum(simulation.state["buffer"])
    simulation.state["buffer_history"].append((simulation.current_time, current_buffer_size))
    
    # If the link is idle, start transmission
    if simulation.state["link_state"] == "idle":
        simulation.state["link_state"] = "busy"
        simulation.schedule_event(
            time=simulation.current_time,
            action=start_transmission,
            data=None
        )
    
    # Schedule the next packet arrival
    interarrival_time = random.expovariate(1.0 / data["mean_interarrival"])
    next_arrival = simulation.current_time + interarrival_time
    
    if next_arrival < simulation.max_time:
        simulation.schedule_event(
            time=next_arrival,
            action=packet_arrival,
            data=data
        )

# Start transmission event
def start_transmission(simulation, data):
    # If there are packets in the buffer, transmit the next one
    if simulation.state["buffer"]:
        packet_size = simulation.state["buffer"].pop(0)
        
        # Calculate transmission time based on link speed
        transmission_time = packet_size / simulation.state["link_speed"]
        
        # Schedule transmission completion
        simulation.schedule_event(
            time=simulation.current_time + transmission_time,
            action=end_transmission,
            data={"packet_size": packet_size}
        )
        
        # Update buffer size history
        current_buffer_size = sum(simulation.state["buffer"])
        simulation.state["buffer_history"].append((simulation.current_time, current_buffer_size))
    else:
        # No packets to transmit, link becomes idle
        simulation.state["link_state"] = "idle"

# End transmission event
def end_transmission(simulation, data):
    # Record statistics
    simulation.state["transmitted_bytes"] += data["packet_size"]
    simulation.state["transmitted_packets"] += 1
    
    # Start the next transmission if there are more packets
    if simulation.state["buffer"]:
        simulation.schedule_event(
            time=simulation.current_time,
            action=start_transmission,
            data=None
        )
    else:
        # No more packets, link becomes idle
        simulation.state["link_state"] = "idle"

# Create and run network simulation
def run_network_simulation(mean_interarrival=0.1, mean_packet_size=1000, link_speed=10000, simulation_time=100):
    # Initial state
    initial_state = {
        "buffer": [],                  # Packet buffer
        "link_state": "idle",          # Link state (idle or busy)
        "total_packets": 0,            # Total packets arrived
        "transmitted_packets": 0,      # Total packets transmitted
        "transmitted_bytes": 0,        # Total bytes transmitted
        "link_speed": link_speed,      # Link speed in bytes per second
        "buffer_history": [],          # Buffer size history
        "value": 0                     # Required for the base simulation
    }
    
    # Initial events
    initial_events = [
        (0.0, packet_arrival, {"mean_interarrival": mean_interarrival, "mean_packet_size": mean_packet_size})
    ]
    
    # Create and run the simulation
    sim = SimulatorRegistry.create(
        "DiscreteEvent",
        max_time=simulation_time,
        initial_events=initial_events,
        time_step=0.1,
        random_seed=42
    )
    
    # Set the initial state
    sim.state = initial_state
    
    # Run the simulation
    sim.run_simulation()
    
    return sim

# Run the network simulation
network_sim = run_network_simulation(
    mean_interarrival=0.1,     # Mean time between packet arrivals (seconds)
    mean_packet_size=1000,     # Mean packet size (bytes)
    link_speed=10000,          # Link speed (bytes per second)
    simulation_time=100        # Simulation time (seconds)
)

# Extract and analyze results
total_packets = network_sim.state["total_packets"]
transmitted_packets = network_sim.state["transmitted_packets"]
transmitted_bytes = network_sim.state["transmitted_bytes"]
throughput = transmitted_bytes / network_sim.max_time

print(f"Total packets: {total_packets}")
print(f"Transmitted packets: {transmitted_packets}")
print(f"Throughput: {throughput:.2f} bytes/second")

# Plot buffer size over time
times, buffer_sizes = zip(*network_sim.state["buffer_history"])
plt.figure(figsize=(12, 6))
plt.step(times, buffer_sizes, where='post')
plt.title('Network Buffer Size Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Buffer Size (bytes)')
plt.grid(True)
plt.show()
```

## Advanced Topics

### Event Priorities

Events with the same time are processed in order of priority:

```python
# High priority event (priority=0, the default)
simulation.schedule_event(
    time=10.0,
    action=high_priority_action,
    priority=0,
    data=None
)

# Low priority event (higher number = lower priority)
simulation.schedule_event(
    time=10.0,
    action=low_priority_action,
    priority=1,
    data=None
)
```

### Custom State Variables

You can add custom variables to the simulation state:

```python
def initialize_simulation(simulation, data):
    # Set up custom state variables
    simulation.state["customers"] = []
    simulation.state["resources"] = {"tellers": 3, "managers": 1}
    simulation.state["statistics"] = {"wait_times": [], "service_times": []}
```

### Stochastic Processes

You can model stochastic processes using random distributions:

```python
def exponential_event(simulation, data):
    # Generate random time from exponential distribution
    mean_time = data["mean"]
    random_time = random.expovariate(1.0 / mean_time)
    
    # Schedule next event
    next_time = simulation.current_time + random_time
    simulation.schedule_event(
        time=next_time,
        action=exponential_event,
        data=data
    )
```

## API Reference

### DiscreteEventSimulation Class

```python
DiscreteEventSimulation(
    max_time: float,
    initial_events: List[Tuple[float, Callable, Any]] = None,
    time_step: float = 1.0,
    random_seed: Optional[int] = None
)
```

#### Methods

- **schedule_event(time, action, priority=0, data=None)**: Schedule a new event
- **run_simulation()**: Run the simulation and return results
- **reset()**: Reset the simulation to its initial state

### Event Class

```python
Event(
    time: float,
    action: Callable,
    priority: int = 0,
    data: Any = None
)
```

## Further Reading

For more information about discrete event simulation:

- [Introduction to Discrete-Event Simulation](https://en.wikipedia.org/wiki/Discrete_event_simulation)
- [Simulation Modeling and Analysis](https://www.amazon.com/Simulation-Modeling-Analysis-Averill-Law/dp/0073401323) by Averill Law
- [SimPy](https://simpy.readthedocs.io/) - A Python framework for discrete event simulation