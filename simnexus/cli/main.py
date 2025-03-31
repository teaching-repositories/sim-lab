"""Main entry point for SimNexus CLI."""

import typer
from typing import Optional
import importlib.metadata
import os
import sys
from pathlib import Path

from ..core import (
    StockMarketSimulation,
    ResourceFluctuationsSimulation,
    ProductPopularitySimulation
)

# Create the main Typer app
app = typer.Typer(
    name="simnexus",
    help="SimNexus: Business Simulation Toolkit",
    add_completion=False,
)

# Create simulation commands group
sim_app = typer.Typer(help="Run simulations")
app.add_typer(sim_app, name="sim")

# Create interfaces commands group
ui_app = typer.Typer(help="Launch user interfaces")
app.add_typer(ui_app, name="ui")

# Create utility commands group
util_app = typer.Typer(help="Utility commands")
app.add_typer(util_app, name="util")

# Each simulation type is a subcommand of 'sim'
stock_app = typer.Typer(help="Stock market simulation commands")
resource_app = typer.Typer(help="Resource fluctuations simulation commands")
product_app = typer.Typer(help="Product popularity simulation commands")

sim_app.add_typer(stock_app, name="stock")
sim_app.add_typer(resource_app, name="resource")
sim_app.add_typer(product_app, name="product")


@app.callback()
def main(version: bool = typer.Option(False, "--version", "-v", help="Show version and exit")) -> None:
    """SimNexus: Business Simulation Toolkit for educational use."""
    if version:
        try:
            version = importlib.metadata.version("simnexus")
            typer.echo(f"SimNexus version: {version}")
        except importlib.metadata.PackageNotFoundError:
            typer.echo("SimNexus version: unknown")
        raise typer.Exit()


@stock_app.command("run")
def run_stock_simulation(
    start_price: float = typer.Option(100.0, help="Initial stock price"),
    days: int = typer.Option(365, help="Number of days to simulate"),
    volatility: float = typer.Option(0.02, help="Daily price volatility"),
    drift: float = typer.Option(0.001, help="Daily price drift"),
    event_day: Optional[int] = typer.Option(None, help="Day of market event (optional)"),
    event_impact: float = typer.Option(0.0, help="Impact of market event"),
    output: str = typer.Option("prices.csv", help="Output file for results"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    visualize: bool = typer.Option(False, "--viz", help="Visualize results after simulation"),
):
    """Run a stock market simulation with the given parameters."""
    typer.echo(f"Running stock market simulation for {days} days...")
    
    # Create and run the simulation
    sim = StockMarketSimulation(
        start_price=start_price,
        days=days,
        volatility=volatility,
        drift=drift,
        event_day=event_day,
        event_impact=event_impact,
        random_seed=seed,
    )
    
    prices = sim.run_simulation()
    
    # Save results to CSV
    import csv
    with open(output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Day', 'Price'])
        for day, price in enumerate(prices):
            writer.writerow([day, price])
    
    typer.echo(f"Simulation complete! Results saved to {output}")
    
    # Optional visualization
    if visualize:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(prices)
            
            if event_day is not None:
                plt.axvline(x=event_day, color='red', linestyle='--', 
                           label=f'Market Event (Impact: {event_impact})')
            
            plt.xlabel('Days')
            plt.ylabel('Price ($)')
            plt.title('Stock Market Simulation')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            typer.echo("Visualization requires matplotlib. Install with 'pip install matplotlib'")


@resource_app.command("run")
def run_resource_simulation(
    start_price: float = typer.Option(100.0, help="Initial resource price"),
    days: int = typer.Option(365, help="Number of days to simulate"),
    volatility: float = typer.Option(0.03, help="Daily price volatility"),
    drift: float = typer.Option(0.001, help="Daily price drift"),
    disruption_day: Optional[int] = typer.Option(None, help="Day of supply disruption (optional)"),
    disruption_severity: float = typer.Option(0.0, help="Severity of supply disruption"),
    output: str = typer.Option("resource_prices.csv", help="Output file for results"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    visualize: bool = typer.Option(False, "--viz", help="Visualize results after simulation"),
):
    """Run a resource fluctuations simulation with the given parameters."""
    typer.echo(f"Running resource fluctuations simulation for {days} days...")
    
    # Create and run the simulation
    sim = ResourceFluctuationsSimulation(
        start_price=start_price,
        days=days,
        volatility=volatility,
        drift=drift,
        supply_disruption_day=disruption_day,
        disruption_severity=disruption_severity,
        random_seed=seed,
    )
    
    prices = sim.run_simulation()
    
    # Save results to CSV
    import csv
    with open(output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Day', 'Price'])
        for day, price in enumerate(prices):
            writer.writerow([day, price])
    
    typer.echo(f"Simulation complete! Results saved to {output}")
    
    # Optional visualization
    if visualize:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(prices)
            
            if disruption_day is not None:
                plt.axvline(x=disruption_day, color='red', linestyle='--', 
                           label=f'Supply Disruption (Severity: {disruption_severity})')
            
            plt.xlabel('Days')
            plt.ylabel('Price ($)')
            plt.title('Resource Fluctuations Simulation')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            typer.echo("Visualization requires matplotlib. Install with 'pip install matplotlib'")


@product_app.command("run")
def run_product_simulation():
    """Run a product popularity simulation."""
    typer.echo("Product popularity simulation coming soon!")


@ui_app.command("web")
def run_web(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Launch the web interface."""
    try:
        from ..web.app import run_web_server
        typer.echo(f"Starting web server at http://{host}:{port}")
        run_web_server(host=host, port=port, reload=reload)
    except ImportError:
        typer.echo("Web interface requires additional dependencies.")
        typer.echo("Install with: pip install simnexus[web]")


@ui_app.command("tui")
def run_tui():
    """Launch the terminal user interface."""
    try:
        from ..tui.app import run_tui
        typer.echo("Starting terminal interface...")
        run_tui()
    except ImportError:
        typer.echo("Terminal UI requires additional dependencies.")
        typer.echo("Install with: pip install simnexus[dev]")


@util_app.command("info")
def show_info():
    """Show information about SimNexus."""
    try:
        version = importlib.metadata.version("simnexus")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    typer.echo("SimNexus Information")
    typer.echo("===================")
    typer.echo(f"Version: {version}")
    typer.echo(f"Python: {python_version}")
    typer.echo(f"Platform: {sys.platform}")
    typer.echo(f"Executable: {sys.executable}")
    

if __name__ == "__main__":
    app()