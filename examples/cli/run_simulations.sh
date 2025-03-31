#!/bin/bash
# Example script demonstrating various SimNexus CLI commands

echo "SimNexus CLI Examples"
echo "===================="
echo ""

# Check if SimNexus is installed
if ! command -v simnexus &> /dev/null; then
    echo "SimNexus is not installed. Install with:"
    echo "pip install simnexus[cli]"
    exit 1
fi

# Get version info
echo "Getting SimNexus version..."
simnexus --version
echo ""

# Display system info
echo "Getting system information..."
simnexus util info
echo ""

# Run a stock market simulation
echo "Running a stock market simulation..."
simnexus sim stock run \
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
simnexus sim resource run \
    --start-price 100 \
    --days 250 \
    --volatility 0.015 \
    --drift 0.0003 \
    --disruption-day 100 \
    --disruption-severity 0.3 \
    --output resource_simulation.csv
echo ""

# Run a basic simulation with visualization (if matplotlib is installed)
echo "Running a simulation with visualization..."
simnexus sim stock run --days 180 --event-day 90 --event-impact -0.15 --viz
echo ""

echo "Examples complete. Results saved to:"
echo "- stock_simulation.csv"
echo "- resource_simulation.csv"