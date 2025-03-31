# SimNexus Examples

This directory contains example code showing how to use SimNexus in different contexts.

## Directory Structure

- `python/`: Basic Python script examples for using SimNexus as a library
- `cli/`: Command-line interface examples showing how to use the CLI tools
- `jupyter/`: Jupyter notebook examples for interactive data analysis

## Python Examples

The Python examples demonstrate how to use SimNexus as an imported library:

- `stock_market_example.py`: Stock market simulation example
- `resource_fluctuations_example.py`: Resource price fluctuation example

To run these examples:

```bash
cd python
python stock_market_example.py
python resource_fluctuations_example.py
```

## CLI Examples

The CLI examples show how to use SimNexus from the command line:

- `run_simulations.sh`: Script demonstrating various CLI commands

To run the CLI examples:

```bash
cd cli
chmod +x run_simulations.sh
./run_simulations.sh
```

## Jupyter Examples

The Jupyter notebook examples demonstrate more complex scenarios and data analysis:

- `simulation_examples.ipynb`: Comprehensive notebook with multiple simulations and analysis

To run the Jupyter examples:

```bash
cd jupyter
jupyter notebook simulation_examples.ipynb
```

## Requirements

These examples require SimNexus to be installed with the appropriate dependencies:

```bash
# For Python examples
pip install simnexus

# For CLI examples
pip install simnexus[cli]

# For Jupyter examples
pip install simnexus[dev] pandas jupyter
```