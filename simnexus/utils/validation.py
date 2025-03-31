"""Validation utilities for SimNexus."""

from typing import Dict, Any, List, Tuple, Optional


def validate_simulation_params(params: Dict[str, Any], simulation_type: str) -> Tuple[bool, List[str]]:
    """
    Validate simulation parameters for a given simulation type.
    
    Args:
        params: The parameters to validate
        simulation_type: The type of simulation ('stock', 'resource', or 'product')
        
    Returns:
        Tuple[bool, List[str]]: A tuple containing validation status and error messages
    """
    errors = []
    
    # Common validations
    if 'days' in params:
        days = params['days']
        if not isinstance(days, int) or days <= 0:
            errors.append(f"Days must be a positive integer, got {days}")
    
    if 'random_seed' in params and params['random_seed'] is not None:
        seed = params['random_seed']
        if not isinstance(seed, int):
            errors.append(f"Random seed must be an integer, got {seed}")
    
    # Simulation-specific validations
    if simulation_type in ['stock', 'resource']:
        if 'start_price' in params:
            start_price = params['start_price']
            if not isinstance(start_price, (int, float)) or start_price <= 0:
                errors.append(f"Start price must be a positive number, got {start_price}")
        
        if 'volatility' in params:
            volatility = params['volatility']
            if not isinstance(volatility, (int, float)) or volatility < 0:
                errors.append(f"Volatility must be a non-negative number, got {volatility}")
        
        if 'drift' in params:
            drift = params['drift']
            if not isinstance(drift, (int, float)):
                errors.append(f"Drift must be a number, got {drift}")
    
    if simulation_type == 'stock':
        if 'event_day' in params and params['event_day'] is not None:
            event_day = params['event_day']
            if not isinstance(event_day, int) or event_day < 0:
                errors.append(f"Event day must be a non-negative integer, got {event_day}")
            
            if 'days' in params and event_day >= params['days']:
                errors.append(f"Event day ({event_day}) must be less than total days ({params['days']})")
    
    if simulation_type == 'resource':
        if 'supply_disruption_day' in params and params['supply_disruption_day'] is not None:
            disruption_day = params['supply_disruption_day']
            if not isinstance(disruption_day, int) or disruption_day < 0:
                errors.append(f"Supply disruption day must be a non-negative integer, got {disruption_day}")
            
            if 'days' in params and disruption_day >= params['days']:
                errors.append(f"Supply disruption day ({disruption_day}) must be less than total days ({params['days']})")
    
    if simulation_type == 'product':
        if 'initial_popularity' in params:
            initial_popularity = params['initial_popularity']
            if not isinstance(initial_popularity, (int, float)) or not 0 <= initial_popularity <= 1:
                errors.append(f"Initial popularity must be between 0 and 1, got {initial_popularity}")
        
        if 'virality_factor' in params:
            virality_factor = params['virality_factor']
            if not isinstance(virality_factor, (int, float)) or virality_factor < 0:
                errors.append(f"Virality factor must be a non-negative number, got {virality_factor}")
        
        if 'marketing_effectiveness' in params:
            marketing_effectiveness = params['marketing_effectiveness']
            if not isinstance(marketing_effectiveness, (int, float)) or not 0 <= marketing_effectiveness <= 1:
                errors.append(f"Marketing effectiveness must be between 0 and 1, got {marketing_effectiveness}")
    
    return len(errors) == 0, errors