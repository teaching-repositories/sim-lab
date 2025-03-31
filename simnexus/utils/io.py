"""I/O utilities for SimNexus."""

import csv
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def save_to_csv(data: List[float], file_path: str, headers: Optional[List[str]] = None) -> None:
    """
    Save simulation data to a CSV file.
    
    Args:
        data: The data points to save
        file_path: Path to the output file
        headers: Optional column headers (defaults to ['Day', 'Value'])
    """
    if headers is None:
        headers = ['Day', 'Value']
    
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for i, value in enumerate(data):
            writer.writerow([i, value])


def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save simulation data to a JSON file.
    
    Args:
        data: The data to save
        file_path: Path to the output file
    """
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=2)


def load_from_csv(file_path: str, value_column: int = 1, skip_header: bool = True) -> List[float]:
    """
    Load simulation data from a CSV file.
    
    Args:
        file_path: Path to the input file
        value_column: Column index for the values (default is 1)
        skip_header: Whether to skip the header row
        
    Returns:
        List[float]: The loaded data points
    """
    data = []
    
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Skip header if needed
        if skip_header:
            next(reader, None)
        
        for row in reader:
            if len(row) > value_column:
                try:
                    data.append(float(row[value_column]))
                except ValueError:
                    pass  # Skip non-numeric values
    
    return data


def load_from_json(file_path: str) -> Dict[str, Any]:
    """
    Load simulation data from a JSON file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Dict[str, Any]: The loaded data
    """
    with open(file_path, 'r') as jsonfile:
        return json.load(jsonfile)