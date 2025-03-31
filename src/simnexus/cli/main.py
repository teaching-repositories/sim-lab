"""Command line interface for SimNexus."""

import typer
from typing import Optional
import importlib.metadata
import os
import sys

from rich.console import Console
from rich.panel import Panel

# Create Typer app with command groups
app = typer.Typer(
    name="simnexus",
    help="Business simulation toolkit for educational use",
    add_completion=False,
)

sim_app = typer.Typer(help="Run various simulations")
ui_app = typer.Typer(help="Launch different user interfaces")
util_app = typer.Typer(help="Utility commands")

app.add_typer(sim_app, name="sim")
app.add_typer(ui_app, name="ui")
app.add_typer(util_app, name="util")

# Create subgroups for simulations
stock_app = typer.Typer(help="Stock market simulation commands")
resource_app = typer.Typer(help="Resource fluctuations simulation commands")
product_app = typer.Typer(help="Product popularity simulation commands")

sim_app.add_typer(stock_app, name="stock")
sim_app.add_typer(resource_app, name="resource")
sim_app.add_typer(product_app, name="product")

console = Console()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show the application version and exit."
    ),
):
    """SimNexus CLI - Business simulation toolkit for educational use."""
    if version:
        try:
            version = importlib.metadata.version("simnexus")
            console.print(f"SimNexus version: {version}")
        except importlib.metadata.PackageNotFoundError:
            console.print("SimNexus version: [italic]development[/italic]")
        raise typer.Exit()


@stock_app.command("run")
def run_stock_simulation(
    days: int = typer.Option(365, help="Number of days to simulate"),
    start_price: float = typer.Option(100.0, help="Starting stock price"),
    volatility: float = typer.Option(0.02, help="Daily volatility (0.01-0.10)"),
    drift: float = typer.Option(0.001, help="Daily price trend"),
    event_day: Optional[int] = typer.Option(None, help="Day of market event (optional)"),
    event_impact: float = typer.Option(0.0, help="Impact of market event (-0.9 to 0.9)"),
    random_seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    output: Optional[str] = typer.Option(None, help="Output CSV file path"),
    viz: bool = typer.Option(False, help="Visualize the results"),
):
    """Run a stock market simulation with the specified parameters."""
    from simnexus import StockMarketSimulation
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    console.print(Panel("Running Stock Market Simulation", style="green"))
    
    # Run simulation
    sim = StockMarketSimulation(
        start_price=start_price,
        days=days,
        volatility=volatility,
        drift=drift,
        event_day=event_day,
        event_impact=event_impact,
        random_seed=random_seed,
    )
    
    prices = sim.run_simulation()
    
    # Display results summary
    console.print(f"Simulation complete: {days} days simulated")
    console.print(f"Starting price: ${start_price:.2f}")
    console.print(f"Final price: ${prices[-1]:.2f}")
    console.print(f"Change: {((prices[-1] / start_price) - 1) * 100:.2f}%")
    
    # Save to CSV if requested
    if output:
        df = pd.DataFrame({"day": range(len(prices)), "price": prices})
        df.to_csv(output, index=False)
        console.print(f"Results saved to {output}")
    
    # Visualize if requested
    if viz:
        plt.figure(figsize=(12, 6))
        plt.plot(prices)
        plt.title("Stock Market Simulation")
        plt.xlabel("Days")
        plt.ylabel("Price ($)")
        
        if event_day is not None:
            plt.axvline(x=event_day, color='red', linestyle='--', 
                       label=f'Market Event (Impact: {event_impact*100:.1f}%)')
            plt.legend()
        
        plt.tight_layout()
        plt.show()


@resource_app.command("run")
def run_resource_simulation(
    days: int = typer.Option(365, help="Number of days to simulate"),
    start_price: float = typer.Option(100.0, help="Starting resource price"),
    volatility: float = typer.Option(0.02, help="Daily volatility (0.01-0.10)"),
    drift: float = typer.Option(0.001, help="Daily price trend"),
    disruption_day: Optional[int] = typer.Option(None, help="Day of supply disruption (optional)"),
    disruption_severity: float = typer.Option(0.0, help="Severity of disruption (0.0-1.0)"),
    random_seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    output: Optional[str] = typer.Option(None, help="Output CSV file path"),
    viz: bool = typer.Option(False, help="Visualize the results"),
):
    """Run a resource fluctuations simulation with the specified parameters."""
    from simnexus import ResourceFluctuationsSimulation
    import matplotlib.pyplot as plt
    import pandas as pd
    
    console.print(Panel("Running Resource Fluctuations Simulation", style="blue"))
    
    # Run simulation
    sim = ResourceFluctuationsSimulation(
        start_price=start_price,
        days=days,
        volatility=volatility,
        drift=drift,
        disruption_day=disruption_day,
        disruption_severity=disruption_severity,
        random_seed=random_seed,
    )
    
    prices = sim.run_simulation()
    
    # Display results summary
    console.print(f"Simulation complete: {days} days simulated")
    console.print(f"Starting price: ${start_price:.2f}")
    console.print(f"Final price: ${prices[-1]:.2f}")
    console.print(f"Change: {((prices[-1] / start_price) - 1) * 100:.2f}%")
    
    # Save to CSV if requested
    if output:
        df = pd.DataFrame({"day": range(len(prices)), "price": prices})
        df.to_csv(output, index=False)
        console.print(f"Results saved to {output}")
    
    # Visualize if requested
    if viz:
        plt.figure(figsize=(12, 6))
        plt.plot(prices)
        plt.title("Resource Price Fluctuations")
        plt.xlabel("Days")
        plt.ylabel("Price ($)")
        
        if disruption_day is not None:
            plt.axvline(x=disruption_day, color='red', linestyle='--', 
                       label=f'Supply Disruption (Severity: {disruption_severity*100:.1f}%)')
            plt.legend()
        
        plt.tight_layout()
        plt.show()


@product_app.command("run")
def run_product_simulation(
    days: int = typer.Option(365, help="Number of days to simulate"),
    initial_popularity: float = typer.Option(0.01, help="Initial popularity (0.0-1.0)"),
    virality: float = typer.Option(0.1, help="Virality factor (0.0-1.0)"),
    marketing: float = typer.Option(0.05, help="Marketing effectiveness (0.0-1.0)"),
    random_seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    output: Optional[str] = typer.Option(None, help="Output CSV file path"),
    viz: bool = typer.Option(False, help="Visualize the results"),
):
    """Run a product popularity simulation with the specified parameters."""
    from simnexus import ProductPopularitySimulation
    import matplotlib.pyplot as plt
    import pandas as pd
    
    console.print(Panel("Running Product Popularity Simulation", style="yellow"))
    
    # Run simulation
    sim = ProductPopularitySimulation(
        days=days,
        initial_popularity=initial_popularity,
        virality_factor=virality,
        marketing_effectiveness=marketing,
        random_seed=random_seed,
    )
    
    popularity = sim.run_simulation()
    
    # Display results summary
    console.print(f"Simulation complete: {days} days simulated")
    console.print(f"Initial popularity: {initial_popularity:.1%}")
    console.print(f"Final popularity: {popularity[-1]:.1%}")
    console.print(f"Peak popularity: {max(popularity):.1%}")
    
    # Save to CSV if requested
    if output:
        df = pd.DataFrame({"day": range(len(popularity)), "popularity": popularity})
        df.to_csv(output, index=False)
        console.print(f"Results saved to {output}")
    
    # Visualize if requested
    if viz:
        plt.figure(figsize=(12, 6))
        plt.plot(popularity)
        plt.title("Product Popularity Over Time")
        plt.xlabel("Days")
        plt.ylabel("Popularity (%)")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()


@ui_app.command("web")
def launch_web(
    host: str = typer.Option("127.0.0.1", help="Host address to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """Launch the web interface."""
    try:
        import uvicorn
        from simnexus.web.app import create_app
        
        console.print(Panel(f"Launching web interface at http://{host}:{port}", style="green"))
        uvicorn.run(create_app, host=host, port=port)
    except ImportError:
        console.print(
            "Web dependencies not installed. Install with: pip install simnexus[web]",
            style="red"
        )
        raise typer.Exit(1)


@ui_app.command("tui")
def launch_tui():
    """Launch the terminal user interface."""
    try:
        from simnexus.tui.app import run_app
        
        console.print(Panel("Launching terminal user interface", style="blue"))
        run_app()
    except ImportError:
        console.print(
            "TUI dependencies not installed. Make sure 'textual' is installed.",
            style="red"
        )
        raise typer.Exit(1)


@util_app.command("info")
def show_info():
    """Show information about the SimNexus package."""
    try:
        version = importlib.metadata.version("simnexus")
        console.print(Panel.fit("SimNexus Information", style="green"))
        console.print(f"Version: {version}")
        console.print(f"Python: {sys.version.split()[0]}")
        console.print(f"Path: {os.path.dirname(os.path.abspath(__file__))}")
    except importlib.metadata.PackageNotFoundError:
        console.print("SimNexus package not installed in development mode.", style="yellow")
        console.print(f"Path: {os.path.dirname(os.path.abspath(__file__))}")


if __name__ == "__main__":
    app()