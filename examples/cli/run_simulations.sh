#!/bin/bash
# Example script demonstrating various SimLab CLI commands

echo "SimLab CLI Examples"
echo "===================="
echo ""

# Check if SimLab is installed
if ! command -v simlab &> /dev/null; then
    echo "SimLab is not installed. Install with:"
    echo "pip install sim-lab[cli]"
    exit 1
fi

# Get version info
echo "Getting SimLab version..."
simlab --version
echo ""

# Display system info
echo "Getting system information..."
simlab util info
echo ""

# Run a stock market simulation
echo "Running a stock market simulation..."
simlab sim stock run \
    --start-price 100 \
    --days 365 \
    --volatility 0.03 \
    --drift -0.001 \
    --event-day 100 \
    --event-impact -0.2 \
    --output stock_simulation.csv
echo ""

# Run a resource fluctuations simulation
echo "Running a resource fluctuations simulation..."
simlab sim resource run \
    --start-price 100 \
    --days 250 \
    --volatility 0.015 \
    --drift 0.0003 \
    --disruption-day 100 \
    --disruption-severity 0.3 \
    --output resource_simulation.csv
echo ""

# Run a product popularity simulation
echo "Running a product popularity simulation..."
simlab sim product run \
    --initial-popularity 0.02 \
    --virality 0.12 \
    --marketing 0.05 \
    --days 200 \
    --output product_simulation.csv
echo ""

# Run a basic simulation with visualization (if matplotlib is installed)
echo "Running a simulation with visualization..."
simlab sim stock run --days 180 --event-day 90 --event-impact -0.15 --viz
echo ""

echo "Examples complete. Results saved to:"
echo "- stock_simulation.csv"
echo "- resource_simulation.csv"
echo "- product_simulation.csv"