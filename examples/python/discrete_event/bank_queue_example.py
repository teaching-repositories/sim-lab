"""
Example of using SimLab to create a bank queueing simulation using the discrete event framework.

This example demonstrates:
1. Setting up a discrete event simulation for a bank with tellers and customers
2. Defining custom event action functions
3. Tracking queue statistics over time
4. Visualizing the results
"""

from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np
import random

# Define the event action functions

def customer_arrival(simulation, data):
    """Handle a customer arrival event."""
    # Increment customer count and add to queue
    simulation.state["total_customers"] += 1
    current_customer = simulation.state["total_customers"]
    arrival_time = simulation.current_time
    
    # Store arrival time for wait time calculation
    simulation.state["arrival_times"][current_customer] = arrival_time
    
    # Add to queue
    simulation.state["queue"].append(current_customer)
    queue_length = len(simulation.state["queue"])
    
    # Update maximum queue length if needed
    simulation.state["max_queue_length"] = max(queue_length, simulation.state["max_queue_length"])
    
    # Record queue length at this time
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


def service_start(simulation, data):
    """Start serving the next customer in queue."""
    # Check if there are customers in queue
    if simulation.state["queue"]:
        # Take the next customer from the queue
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
    else:
        # No customers to serve, teller becomes idle
        simulation.state["idle_tellers"] += 1


def service_completion(simulation, data):
    """Handle service completion for a customer."""
    # Calculate waiting time for this customer
    arrival_time = simulation.state["arrival_times"].get(data["customer"], 0)
    start_service_time = simulation.current_time - data["service_time"]
    wait_time = start_service_time - arrival_time
    
    # Update statistics
    simulation.state["total_wait_time"] += wait_time
    simulation.state["completed_services"] += 1
    simulation.state["wait_times"].append(wait_time)
    
    # Record service completion
    simulation.state["service_completions"].append((simulation.current_time, data["customer"]))
    
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


# Main simulation function
def run_bank_simulation(num_tellers=2, mean_interarrival=5.0, mean_service_time=3.0, simulation_time=480):
    """Run a bank simulation for specified parameters.
    
    Args:
        num_tellers: Number of tellers working at the bank
        mean_interarrival: Mean time between customer arrivals (minutes)
        mean_service_time: Mean time to serve a customer (minutes)
        simulation_time: Total simulation time (minutes)
        
    Returns:
        The completed simulation
    """
    # Set up initial state
    initial_state = {
        "queue": [],                    # Customer queue
        "idle_tellers": num_tellers,    # Initially all tellers are idle
        "total_customers": 0,           # Counter for arriving customers
        "completed_services": 0,        # Counter for completed services
        "total_wait_time": 0,           # Total wait time for all customers
        "max_queue_length": 0,          # Maximum queue length observed
        "mean_service_time": mean_service_time,  # Mean service time
        "arrival_times": {},            # Tracks when each customer arrived
        "queue_length_history": [],     # History of queue lengths over time
        "wait_times": [],               # List of individual wait times
        "service_completions": [],      # When services were completed
        "value": 0                      # Required by the simulation framework
    }
    
    # Set up initial events - first customer arrival
    initial_events = [
        (0.0, customer_arrival, {"mean_interarrival": mean_interarrival})
    ]
    
    # Create the simulation
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


# Run a simulation and analyze the results
if __name__ == "__main__":
    # Run the simulation
    print("Running bank queue simulation...")
    bank_sim = run_bank_simulation(
        num_tellers=3,
        mean_interarrival=2.0,  # Customer arrives every 2 minutes on average
        mean_service_time=5.0,  # Takes 5 minutes to serve a customer on average
        simulation_time=480     # 8-hour work day (480 minutes)
    )
    
    # Extract results
    state = bank_sim.state
    total_customers = state["total_customers"]
    completed_services = state["completed_services"]
    remaining_customers = len(state["queue"]) + (3 - state["idle_tellers"])
    
    # Calculate statistics
    if completed_services > 0:
        avg_wait_time = state["total_wait_time"] / completed_services
        max_wait_time = max(state["wait_times"]) if state["wait_times"] else 0
    else:
        avg_wait_time = 0
        max_wait_time = 0
    
    # Print summary statistics
    print("\nSimulation Results:")
    print(f"Total customers arrived: {total_customers}")
    print(f"Customers served: {completed_services}")
    print(f"Customers remaining at close: {remaining_customers}")
    print(f"Maximum queue length: {state['max_queue_length']}")
    print(f"Average wait time: {avg_wait_time:.2f} minutes")
    print(f"Maximum wait time: {max_wait_time:.2f} minutes")
    
    # Visualize the queue length over time
    if state["queue_length_history"]:
        times, queue_lengths = zip(*state["queue_length_history"])
        
        plt.figure(figsize=(12, 6))
        plt.step(times, queue_lengths, where='post')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Queue Length')
        plt.title('Bank Queue Length Throughout the Day')
        plt.grid(True, alpha=0.3)
        
        # Add markers for every hour
        hours = range(60, 480, 60)
        for hour in hours:
            plt.axvline(x=hour, color='gray', linestyle='--', alpha=0.5)
            plt.text(hour+5, max(queue_lengths)/2, f"{hour//60}h", alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Plot wait time distribution
        if state["wait_times"]:
            plt.figure(figsize=(10, 6))
            plt.hist(state["wait_times"], bins=20, edgecolor='black', alpha=0.7)
            plt.xlabel('Wait Time (minutes)')
            plt.ylabel('Number of Customers')
            plt.title('Distribution of Customer Wait Times')
            plt.grid(True, alpha=0.3)
            plt.axvline(x=avg_wait_time, color='red', linestyle='--', label=f'Average: {avg_wait_time:.2f} min')
            plt.legend()
            plt.tight_layout()
            plt.show()