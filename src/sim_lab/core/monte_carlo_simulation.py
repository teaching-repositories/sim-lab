"""Monte Carlo Simulation implementation."""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


@SimulatorRegistry.register("MonteCarlo")
class MonteCarloSimulation(BaseSimulation):
    """A simulation class for Monte Carlo methods.
    
    This simulation runs repeated random sampling to obtain numerical results.
    It's useful for risk analysis, optimization, and numerical integration.
    
    Attributes:
        sample_function (Callable): Function that generates random samples.
        evaluation_function (Callable): Function that evaluates each sample.
        num_samples (int): Number of samples to generate per step.
        days (int): Number of steps to simulate (each step generates num_samples).
        random_seed (Optional[int]): Seed for random number generation.
    """
    
    def __init__(
        self, 
        sample_function: Callable[[], Any],
        evaluation_function: Callable[[Any], float],
        num_samples: int = 1000,
        days: int = 100,
        confidence_interval: bool = True,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the Monte Carlo simulation.
        
        Args:
            sample_function: Function that generates random samples.
            evaluation_function: Function that evaluates each sample to a numeric value.
            num_samples: Number of samples to generate per step.
            days: Number of steps to simulate.
            confidence_interval: Whether to calculate confidence intervals.
            random_seed: Seed for random number generation.
        """
        super().__init__(days=days, random_seed=random_seed)
        self.sample_function = sample_function
        self.evaluation_function = evaluation_function
        self.num_samples = num_samples
        self.confidence_interval = confidence_interval
        self.results_history = []
        self.confidence_intervals = []
    
    def run_simulation(self) -> List[float]:
        """Run the Monte Carlo simulation.
        
        For each day, generates multiple samples, evaluates them, and calculates statistics.
        
        Returns:
            A list of mean values for each day's samples.
        """
        self.reset()
        
        results = []
        self.results_history = []
        self.confidence_intervals = []
        
        for _ in range(self.days):
            # Generate samples
            samples = [self.sample_function() for _ in range(self.num_samples)]
            
            # Evaluate samples
            evaluations = [self.evaluation_function(sample) for sample in samples]
            self.results_history.append(evaluations)
            
            # Calculate mean
            mean_value = np.mean(evaluations)
            results.append(mean_value)
            
            # Calculate confidence interval if requested
            if self.confidence_interval:
                std_dev = np.std(evaluations, ddof=1)  # Sample standard deviation
                margin_error = 1.96 * std_dev / np.sqrt(self.num_samples)  # 95% confidence
                self.confidence_intervals.append((mean_value - margin_error, mean_value + margin_error))
        
        return results
    
    def get_confidence_intervals(self) -> List[Tuple[float, float]]:
        """Get the confidence intervals for each day's results.
        
        Returns:
            A list of (lower_bound, upper_bound) tuples representing the 95% confidence interval.
        """
        if not self.confidence_interval:
            raise ValueError("Confidence intervals were not calculated during simulation")
        return self.confidence_intervals
    
    def get_convergence_analysis(self) -> Dict[str, List[float]]:
        """Analyze how the simulation results converge as sample size increases.
        
        Returns:
            A dictionary with sample sizes and the corresponding means at those sizes.
        """
        if not self.results_history:
            raise ValueError("No simulation results available. Run the simulation first.")
        
        # Use the final day's results for convergence analysis
        final_results = self.results_history[-1]
        
        # Calculate running means at different sample sizes
        sample_sizes = [10, 50, 100, 500, 1000, 5000]
        sample_sizes = [s for s in sample_sizes if s <= self.num_samples]
        if self.num_samples not in sample_sizes:
            sample_sizes.append(self.num_samples)
        
        running_means = [np.mean(final_results[:size]) for size in sample_sizes]
        
        return {
            "sample_sizes": sample_sizes,
            "means": running_means
        }
    
    def get_statistics(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Get summary statistics from the simulation.
        
        Returns:
            A dictionary containing statistics such as overall mean, standard deviation,
            and various percentiles.
        """
        if not self.results_history:
            raise ValueError("No simulation results available. Run the simulation first.")
        
        # Flatten all results
        all_results = [item for sublist in self.results_history for item in sublist]
        
        stats = {
            "mean": np.mean(all_results),
            "std_dev": np.std(all_results, ddof=1),
            "min": np.min(all_results),
            "max": np.max(all_results),
            "percentiles": {
                "25th": np.percentile(all_results, 25),
                "50th": np.percentile(all_results, 50),  # median
                "75th": np.percentile(all_results, 75),
                "95th": np.percentile(all_results, 95),
                "99th": np.percentile(all_results, 99)
            }
        }
        
        return stats
    
    @classmethod
    def get_parameters_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about the parameters required by this simulation.
        
        Returns:
            A dictionary mapping parameter names to their metadata.
        """
        # Get base parameters from parent class
        params = super().get_parameters_info()
        
        # Add class-specific parameters
        params.update({
            'sample_function': {
                'type': 'Callable[[], Any]',
                'description': 'Function that generates random samples',
                'required': True
            },
            'evaluation_function': {
                'type': 'Callable[[Any], float]',
                'description': 'Function that evaluates each sample to a numeric value',
                'required': True
            },
            'num_samples': {
                'type': 'int',
                'description': 'Number of samples to generate per step',
                'required': False,
                'default': 1000
            },
            'confidence_interval': {
                'type': 'bool',
                'description': 'Whether to calculate confidence intervals',
                'required': False,
                'default': True
            }
        })
        
        return params