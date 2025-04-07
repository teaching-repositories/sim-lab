"""Discrete Event Simulation implementation."""

import heapq
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


class Event:
    """Represents a discrete event in the simulation.
    
    Attributes:
        time (float): The time at which the event occurs.
        action (Callable): The function to execute when the event occurs.
        priority (int): Lower values indicate higher priority.
        data (Any): Additional data associated with the event.
    """
    
    def __init__(self, time: float, action: Callable, priority: int = 0, data: Any = None) -> None:
        """Initialize a new event.
        
        Args:
            time: The time at which the event occurs.
            action: The function to execute when the event occurs.
            priority: The priority of the event (lower is higher priority).
            data: Additional data associated with the event.
        """
        self.time = time
        self.action = action
        self.priority = priority
        self.data = data
    
    def __lt__(self, other: 'Event') -> bool:
        """Compare events by time, then by priority."""
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time


@SimulatorRegistry.register("DiscreteEvent")
class DiscreteEventSimulation(BaseSimulation):
    """A simulation class for discrete event simulations.
    
    This simulation processes events in chronological order, with each event potentially
    generating new events. The simulation runs until a specified end time or until
    there are no more events to process.
    
    Attributes:
        max_time (float): The maximum simulation time.
        days (int): Used for compatibility with other simulations (days = max_time).
        current_time (float): The current simulation time.
        event_queue (List[Event]): The priority queue of pending events.
        state (Dict[str, Any]): The current state of the simulation.
        results (List[float]): The results of the simulation at each time step.
        random_seed (Optional[int]): Seed for random number generation.
    """
    
    def __init__(
        self, max_time: float, initial_events: List[Tuple[float, Callable, Any]] = None,
        time_step: float = 1.0, random_seed: Optional[int] = None
    ) -> None:
        """Initialize the discrete event simulation.
        
        Args:
            max_time: The maximum simulation time.
            initial_events: List of (time, action, data) tuples to initialize the event queue.
            time_step: The time step for recording results (default: 1.0).
            random_seed: Seed for random number generation.
        """
        # Convert max_time to days for BaseSimulation compatibility
        days = int(max_time)
        super().__init__(days=days, random_seed=random_seed)
        
        self.max_time = max_time
        self.current_time = 0.0
        self.event_queue = []
        self.state = {"value": 0.0}  # Default state with a value field
        self.results = [0.0]  # Start with initial value
        self.time_step = time_step
        
        # Schedule initial events
        if initial_events:
            for time, action, data in initial_events:
                self.schedule_event(time, action, data=data)
    
    def schedule_event(self, time: float, action: Callable, priority: int = 0, data: Any = None) -> None:
        """Schedule a new event to occur at the specified time.
        
        Args:
            time: The absolute time at which the event should occur.
            action: The function to execute when the event occurs.
            priority: The priority of the event (lower is higher priority).
            data: Additional data associated with the event.
        """
        event = Event(time, action, priority, data)
        heapq.heappush(self.event_queue, event)
    
    def run_simulation(self) -> List[float]:
        """Run the simulation until max_time or until there are no more events.
        
        The simulation processes events in chronological order, with each event potentially
        generating new events by calling schedule_event().
        
        Returns:
            A list of values representing the simulation state at regular intervals.
        """
        # Reset the simulation
        self.reset()
        
        next_recording_time = self.time_step
        
        # Process events until max_time or until there are no more events
        while self.event_queue and self.current_time < self.max_time:
            # Get the next event
            event = heapq.heappop(self.event_queue)
            
            # Update the current time
            self.current_time = event.time
            
            # Record results at regular intervals
            while next_recording_time <= self.current_time and next_recording_time <= self.max_time:
                self.results.append(self.state["value"])
                next_recording_time += self.time_step
            
            # Process the event if we haven't exceeded max_time
            if self.current_time <= self.max_time:
                event.action(self, event.data)
        
        # Make sure we have results for all time steps
        while len(self.results) <= self.days:
            self.results.append(self.state["value"])
        
        return self.results
    
    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        super().reset()
        self.current_time = 0.0
        self.state = {"value": 0.0}
        self.results = [0.0]
    
    @classmethod
    def get_parameters_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about the parameters required by this simulation.
        
        Returns:
            A dictionary mapping parameter names to their metadata.
        """
        # Get base parameters from parent class
        params = super().get_parameters_info()
        
        # Replace 'days' with 'max_time'
        del params['days']
        
        # Add class-specific parameters
        params.update({
            'max_time': {
                'type': 'float',
                'description': 'The maximum simulation time',
                'required': True
            },
            'initial_events': {
                'type': 'List[Tuple[float, Callable, Any]]',
                'description': 'List of (time, action, data) tuples to initialize the event queue',
                'required': False,
                'default': []
            },
            'time_step': {
                'type': 'float',
                'description': 'The time step for recording results',
                'required': False,
                'default': 1.0
            }
        })
        
        return params