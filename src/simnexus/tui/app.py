"""Terminal User Interface for SimNexus."""

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Static, Button, Label, Input, Select
from textual.binding import Binding
from textual.screen import Screen

from simnexus import (
    StockMarketSimulation,
    ResourceFluctuationsSimulation,
    ProductPopularitySimulation
)

class WelcomeScreen(Screen):
    """Welcome screen with simulation type selection."""
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Static("# Welcome to SimNexus", classes="title"),
            Static("Select a simulation type to begin:", classes="subtitle"),
            Button("Stock Market Simulation", id="stock", variant="primary"),
            Button("Resource Fluctuations Simulation", id="resource"),
            Button("Product Popularity Simulation", id="product"),
            classes="welcome",
        )
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "stock":
            self.app.push_screen("stock_form")
        elif event.button.id == "resource":
            self.app.push_screen("resource_form")
        elif event.button.id == "product":
            self.app.push_screen("product_form")


class SimNexusApp(App):
    """Main SimNexus TUI application."""
    
    CSS = """
    .welcome {
        layout: vertical;
        content-align: center middle;
        padding: 2 4;
        width: 100%;
        height: 100%;
    }
    
    .welcome .title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    
    .welcome .subtitle {
        text-align: center;
        margin-bottom: 2;
    }
    
    Button {
        width: 40;
        margin: 1 0;
    }
    """
    
    SCREENS = {"welcome": WelcomeScreen}
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "app.pop_screen", "Back", priority=True),
    ]
    
    def on_mount(self) -> None:
        """Event handler called when app is mounted."""
        self.push_screen("welcome")


def run_app():
    """Run the SimNexus TUI application."""
    app = SimNexusApp()
    app.run()