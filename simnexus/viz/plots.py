"""Plotting functions for SimNexus."""

from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np


def plot_time_series(
    data: List[float],
    title: str = "Time Series Plot",
    xlabel: str = "Time",
    ylabel: str = "Value",
    events: Optional[Dict[int, str]] = None,
    backend: str = "matplotlib",
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot a time series of simulation data.
    
    Args:
        data: List of data points to plot
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        events: Dictionary mapping days to event labels
        backend: Plotting backend ('matplotlib' or 'plotly')
        save_path: Optional path to save the plot
        
    Returns:
        The plot object (backend-specific)
    """
    if backend == "matplotlib":
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data)
            
            # Add event markers if provided
            if events:
                for day, label in events.items():
                    if 0 <= day < len(data):
                        ax.axvline(x=day, color='red', linestyle='--')
                        ax.text(day, max(data), label, rotation=90, va='top')
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                
            return fig
            
        except ImportError:
            print("Matplotlib not available. Install with 'pip install matplotlib'")
            return None
            
    elif backend == "plotly":
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=data, mode='lines'))
            
            # Add event markers if provided
            if events:
                for day, label in events.items():
                    if 0 <= day < len(data):
                        fig.add_vline(x=day, line_dash="dash", line_color="red")
                        fig.add_annotation(x=day, y=max(data), text=label, 
                                          showarrow=True, arrowhead=1)
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                template="plotly_white"
            )
            
            if save_path:
                fig.write_image(save_path)
                
            return fig
            
        except ImportError:
            print("Plotly not available. Install with 'pip install plotly'")
            return None
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def plot_distribution(
    data: List[float],
    title: str = "Distribution Plot",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    bins: int = 30,
    backend: str = "matplotlib",
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot the distribution of simulation data.
    
    Args:
        data: List of data points to plot
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        bins: Number of histogram bins
        backend: Plotting backend ('matplotlib' or 'plotly')
        save_path: Optional path to save the plot
        
    Returns:
        The plot object (backend-specific)
    """
    if backend == "matplotlib":
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data, bins=bins, alpha=0.7)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                
            return fig
            
        except ImportError:
            print("Matplotlib not available. Install with 'pip install matplotlib'")
            return None
            
    elif backend == "plotly":
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data, nbinsx=bins))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                template="plotly_white"
            )
            
            if save_path:
                fig.write_image(save_path)
                
            return fig
            
        except ImportError:
            print("Plotly not available. Install with 'pip install plotly'")
            return None
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")