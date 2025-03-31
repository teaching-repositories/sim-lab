"""Textual TUI application for SimNexus."""

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Button, Static, Input, Select
from textual.reactive import reactive
from textual import events
from typing import Dict, Any, List, Optional

from ..core import (
    StockMarketSimulation,
    ResourceFluctuationsSimulation,
    ProductPopularitySimulation
)


class WelcomeScreen(Static):
    """Welcome screen widget."""
    
    def compose(self) -> ComposeResult:
        """Compose the welcome screen."""
        yield Static("# SimNexus: Business Simulation Toolkit", classes="title")
        yield Static("Select a simulation type to begin:", classes="subtitle")
        
        with Container(classes="buttons"):
            yield Button("Stock Market Simulation", id="btn_stock", classes="nav")
            yield Button("Resource Fluctuations Simulation", id="btn_resource", classes="nav")
            yield Button("Product Popularity Simulation", id="btn_product", classes="nav")


class SimulationForm(Static):
    """Base form for simulation parameters."""
    
    simulation_type = reactive("stock")
    params = reactive({})
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_run":
            self.run_simulation()
        elif event.button.id == "btn_back":
            self.app.switch_screen("welcome")


class StockMarketForm(SimulationForm):
    """Form for stock market simulation parameters."""
    
    simulation_type = "stock"
    
    def compose(self) -> ComposeResult:
        """Compose the stock market form."""
        yield Static("# Stock Market Simulation", classes="title")
        
        with Container(classes="form"):
            yield Static("Starting Price:", classes="label")
            yield Input(value="100.0", id="start_price")
            
            yield Static("Days:", classes="label")
            yield Input(value="365", id="days")
            
            yield Static("Volatility:", classes="label")
            yield Input(value="0.02", id="volatility")
            
            yield Static("Drift:", classes="label")
            yield Input(value="0.001", id="drift")
            
            yield Static("Event Day (optional):", classes="label")
            yield Input(value="", id="event_day")
            
            yield Static("Event Impact:", classes="label")
            yield Input(value="0.0", id="event_impact")
        
        with Container(classes="buttons"):
            yield Button("Run Simulation", id="btn_run")
            yield Button("Back", id="btn_back")
    
    def run_simulation(self) -> None:
        """Run the stock market simulation."""
        # Get values from form
        try:
            start_price = float(self.query_one("#start_price").value)
            days = int(self.query_one("#days").value)
            volatility = float(self.query_one("#volatility").value)
            drift = float(self.query_one("#drift").value)
            
            event_day_input = self.query_one("#event_day").value
            event_day = int(event_day_input) if event_day_input else None
            
            event_impact = float(self.query_one("#event_impact").value)
            
            # Create and run simulation
            sim = StockMarketSimulation(
                start_price=start_price,
                days=days,
                volatility=volatility,
                drift=drift,
                event_day=event_day,
                event_impact=event_impact
            )
            
            prices = sim.run_simulation()
            
            # Switch to results screen
            self.app.simulations[self.simulation_type] = {
                "data": prices,
                "params": {
                    "start_price": start_price,
                    "days": days,
                    "volatility": volatility,
                    "drift": drift,
                    "event_day": event_day,
                    "event_impact": event_impact,
                }
            }
            self.app.switch_screen("results")
            
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")


class ResultsScreen(Static):
    """Results screen to display simulation output."""
    
    def compose(self) -> ComposeResult:
        """Compose the results screen."""
        yield Static("# Simulation Results", classes="title")
        yield Static("", id="results_text")
        
        with Container(classes="buttons"):
            yield Button("Save Results", id="btn_save")
            yield Button("New Simulation", id="btn_new")
    
    def on_mount(self) -> None:
        """Handle mount event to update results."""
        sim_type = self.app.current_simulation
        if sim_type and sim_type in self.app.simulations:
            sim_data = self.app.simulations[sim_type]
            
            # Simple text-based visualization
            data = sim_data["data"]
            params = sim_data["params"]
            
            # Create a basic ASCII chart
            max_val = max(data)
            min_val = min(data)
            range_val = max_val - min_val
            
            chart_height = 10
            chart_width = min(80, len(data))
            
            chart = []
            for i in range(chart_height):
                threshold = min_val + (range_val * (chart_height - i - 1) / chart_height)
                line = ""
                for j in range(chart_width):
                    idx = j * len(data) // chart_width
                    if data[idx] >= threshold:
                        line += "█"
                    else:
                        line += " "
                chart.append(line)
            
            # Add axis labels
            chart.append("─" * chart_width)
            
            # Combine everything
            text = f"## {sim_type.title()} Simulation\n\n"
            
            # Parameters
            text += "Parameters:\n"
            for key, value in params.items():
                text += f"- {key}: {value}\n"
            
            text += "\nChart:\n"
            text += f"{max_val:.2f} ┌" + "─" * chart_width + "┐\n"
            for line in chart:
                text += "│" + line + "│\n"
            text += f"{min_val:.2f} └" + "─" * chart_width + "┘\n"
            text += f"    0{' '*(chart_width-8)}{params.get('days', len(data))}\n"
            
            # Update the results text
            self.query_one("#results_text").update(text)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_save":
            self.save_results()
        elif event.button.id == "btn_new":
            self.app.switch_screen("welcome")
    
    def save_results(self) -> None:
        """Save simulation results to a file."""
        import csv
        import os
        from pathlib import Path
        
        sim_type = self.app.current_simulation
        if not sim_type or sim_type not in self.app.simulations:
            self.notify("No simulation results to save", severity="error")
            return
        
        sim_data = self.app.simulations[sim_type]
        
        # Create output directory
        output_dir = Path.home() / "simnexus_results"
        output_dir.mkdir(exist_ok=True)
        
        # Create filename
        filename = f"{sim_type}_simulation_{len(os.listdir(output_dir))}.csv"
        filepath = output_dir / filename
        
        # Save to CSV
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Day', 'Value'])
                for i, value in enumerate(sim_data["data"]):
                    writer.writerow([i, value])
            
            self.notify(f"Results saved to {filepath}", severity="information")
        except Exception as e:
            self.notify(f"Error saving results: {e}", severity="error")


class SimNexusTUI(App):
    """Main SimNexus TUI application."""
    
    CSS = """
    Screen {
        align: center middle;
        background: $surface;
    }
    
    .title {
        dock: top;
        padding: 2 4;
        background: $boost;
        color: $text;
        text-align: center;
        text-style: bold;
        width: 100%;
    }
    
    .subtitle {
        margin: 1 0;
        text-align: center;
    }
    
    .buttons {
        layout: horizontal;
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1;
    }
    
    Button {
        margin: 1 2;
    }
    
    .form {
        width: 100%;
        height: auto;
        padding: 1 2;
        layout: grid;
        grid-size: 2;
        grid-rows: auto;
        grid-columns: 1fr 3fr;
    }
    
    .label {
        text-align: right;
        padding: 1 1;
    }
    
    Input {
        width: 100%;
    }
    
    #results_text {
        width: 100%;
        height: auto;
        padding: 1 2;
        background: $surface;
        color: $text;
    }
    """
    
    SCREENS = {
        "welcome": WelcomeScreen(),
        "stock": StockMarketForm(),
        "results": ResultsScreen(),
    }
    
    def __init__(self) -> None:
        """Initialize the application."""
        super().__init__()
        self.simulations: Dict[str, Dict[str, Any]] = {}
        self.current_simulation: Optional[str] = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Footer()
    
    def on_mount(self) -> None:
        """Handle the app start event."""
        self.switch_screen("welcome")
    
    def switch_screen(self, screen_name: str) -> None:
        """Switch the current screen."""
        if screen_name in self.SCREENS:
            self.current_screen = self.SCREENS[screen_name]
            
            if screen_name in ["stock", "resource", "product"]:
                self.current_simulation = screen_name
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events from the welcome screen."""
        button_id = event.button.id
        if button_id in ["btn_stock", "btn_resource", "btn_product"]:
            screen_name = button_id.split("_")[1]
            self.switch_screen(screen_name)


def run_tui() -> None:
    """Run the SimNexus TUI application."""
    app = SimNexusTUI()
    app.run()


if __name__ == "__main__":
    run_tui()