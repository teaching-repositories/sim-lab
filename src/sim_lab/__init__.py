# Import core simulations for backwards compatibility
from .core import (
    ProductPopularitySimulation,
    ResourceFluctuationsSimulation,
    StockMarketSimulation
)

# Package metadata
__version__ = "0.3.0"
__author__ = "Michael Borck <michael@borck.me>"

# Make core types available at the package level for easy imports
__all__ = [
    "ProductPopularitySimulation",
    "ResourceFluctuationsSimulation",
    "StockMarketSimulation"
]
