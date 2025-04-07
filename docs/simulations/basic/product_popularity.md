# Product Popularity Simulation

The Product Popularity Simulation models the dynamics of product demand over time, incorporating factors like natural growth, marketing impact, and promotional campaigns.

## Overview

This simulation is designed to model product popularity and demand with the following key features:

- Natural growth patterns based on product lifecycle
- Impact of ongoing marketing efforts
- Effect of promotional campaigns or events
- Reproducible results with random seed control

## Basic Usage

```python
from sim_lab.core import SimulatorRegistry

# Create a product popularity simulation
sim = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=100,       # Initial units sold per day
    days=365,               # Simulate for a year
    growth_rate=0.002,      # Natural daily growth rate (0.2%)
    marketing_impact=0.001, # Daily marketing impact (0.1%)
    random_seed=42          # For reproducible results
)

# Run the simulation
demand = sim.run_simulation()

# Access the results
print(f"Initial daily demand: {demand[0]:.0f} units")
print(f"Final daily demand: {demand[-1]:.0f} units")
print(f"Growth over period: {(demand[-1]/demand[0] - 1) * 100:.2f}%")
```

## Parameters

The Product Popularity Simulation accepts the following parameters:

| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `start_demand` | float | Initial demand for the product | Yes | - |
| `days` | int | Duration of the simulation in days | Yes | - |
| `growth_rate` | float | Natural growth rate of product demand | Yes | - |
| `marketing_impact` | float | Impact of ongoing marketing on demand | Yes | - |
| `promotion_day` | int | Day on which a promotional event occurs | No | None |
| `promotion_effectiveness` | float | Effectiveness of the promotion | No | 0 |
| `random_seed` | int | Seed for random number generation | No | None |

## Understanding the Model

### Demand Generation

The product demand for each day is calculated using the following formula:

```
demand[day] = previous_demand * (1 + growth_rate) + previous_demand * marketing_impact
```

This formula has two components:
1. **Natural Growth**: Represented by `previous_demand * (1 + growth_rate)`
2. **Marketing Impact**: Represented by `previous_demand * marketing_impact`

### Promotional Events

If a `promotion_day` is specified, a promotional event occurs on that day with the following impact:

```
demand[promotion_day] = (natural_growth + marketing_influence) * (1 + promotion_effectiveness)
```

This can model events like:
- Product launches
- Major advertising campaigns
- Special promotions or discounts
- Seasonal events like Black Friday

## Example: Modeling a Product Launch Campaign

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Create a simulation with a major launch promotion
sim = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=100,
    days=365,
    growth_rate=0.003,
    marketing_impact=0.002,
    promotion_day=30,        # Promotion after 1 month
    promotion_effectiveness=0.5,  # 50% boost in demand
    random_seed=42
)

# Run the simulation
demand = sim.run_simulation()

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(demand)
plt.axvline(x=30, color='r', linestyle='--', label='Launch Promotion')
plt.title('Product Demand with Launch Promotion')
plt.xlabel('Days')
plt.ylabel('Daily Demand (units)')
plt.legend()
plt.grid(True)
plt.show()
```

## Example: Comparing Marketing Strategies

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Create simulations for different marketing strategies
baseline = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=100,
    days=365,
    growth_rate=0.002,
    marketing_impact=0.001,  # Minimal marketing
    random_seed=42
)

aggressive_marketing = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=100,
    days=365,
    growth_rate=0.002,
    marketing_impact=0.005,  # Heavy ongoing marketing
    random_seed=42
)

promotional_campaign = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=100,
    days=365,
    growth_rate=0.002,
    marketing_impact=0.001,
    promotion_day=90,        # Quarterly promotional event
    promotion_effectiveness=1.0,  # 100% boost from promotion
    random_seed=42
)

# Run simulations
baseline_demand = baseline.run_simulation()
aggressive_demand = aggressive_marketing.run_simulation()
promotional_demand = promotional_campaign.run_simulation()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(baseline_demand, label='Baseline')
plt.plot(aggressive_demand, label='Aggressive Marketing')
plt.plot(promotional_demand, label='Promotional Campaign')
plt.axvline(x=90, color='r', linestyle='--', alpha=0.5, label='Promotion Day')
plt.title('Product Demand Under Different Marketing Strategies')
plt.xlabel('Days')
plt.ylabel('Daily Demand (units)')
plt.legend()
plt.grid(True)
plt.show()

# Compare total sales over the period
baseline_total = sum(baseline_demand)
aggressive_total = sum(aggressive_demand)
promotional_total = sum(promotional_demand)

print(f"Baseline Strategy - Total Units: {baseline_total:.0f}")
print(f"Aggressive Marketing - Total Units: {aggressive_total:.0f} ({(aggressive_total/baseline_total - 1) * 100:.1f}% more)")
print(f"Promotional Campaign - Total Units: {promotional_total:.0f} ({(promotional_total/baseline_total - 1) * 100:.1f}% more)")
```

## Example: Product Lifecycle Model

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np

# Create a simulation that models a full product lifecycle
days = 730  # 2 years

# Introduction phase: slow growth
intro = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=10,
    days=180,
    growth_rate=0.01,      # Steady growth
    marketing_impact=0.005,
    random_seed=42
)

# Growth phase: rapid adoption
growth = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=None,  # Will be set from previous phase
    days=180,
    growth_rate=0.015,      # Faster growth
    marketing_impact=0.01,  # Increased marketing
    random_seed=42
)

# Maturity phase: slowing growth
maturity = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=None,  # Will be set from previous phase
    days=270,
    growth_rate=0.002,      # Slowing growth
    marketing_impact=0.002, # Sustained marketing
    random_seed=42
)

# Decline phase: negative growth
decline = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=None,  # Will be set from previous phase
    days=100,
    growth_rate=-0.005,     # Declining demand
    marketing_impact=0.001, # Reduced marketing
    random_seed=42
)

# Run the introduction phase
intro_demand = intro.run_simulation()

# Run the growth phase starting from where introduction ended
growth.start_demand = intro_demand[-1]
growth_demand = growth.run_simulation()

# Run the maturity phase starting from where growth ended
maturity.start_demand = growth_demand[-1]
maturity_demand = maturity.run_simulation()

# Run the decline phase starting from where maturity ended
decline.start_demand = maturity_demand[-1]
decline_demand = decline.run_simulation()

# Combine the results
full_lifecycle = intro_demand + growth_demand[1:] + maturity_demand[1:] + decline_demand[1:]

# Visualize the product lifecycle
plt.figure(figsize=(14, 7))
plt.plot(full_lifecycle)

# Mark the phases
plt.axvline(x=180, color='g', linestyle='--', label='Introduction → Growth')
plt.axvline(x=360, color='b', linestyle='--', label='Growth → Maturity')
plt.axvline(x=630, color='r', linestyle='--', label='Maturity → Decline')

plt.title('Complete Product Lifecycle')
plt.xlabel('Days')
plt.ylabel('Daily Demand (units)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate phase metrics
intro_total = sum(intro_demand)
growth_total = sum(growth_demand)
maturity_total = sum(maturity_demand)
decline_total = sum(decline_demand)
lifecycle_total = intro_total + growth_total + maturity_total + decline_total

print(f"Introduction Phase: {intro_total:.0f} units ({(intro_total/lifecycle_total)*100:.1f}% of total)")
print(f"Growth Phase: {growth_total:.0f} units ({(growth_total/lifecycle_total)*100:.1f}% of total)")
print(f"Maturity Phase: {maturity_total:.0f} units ({(maturity_total/lifecycle_total)*100:.1f}% of total)")
print(f"Decline Phase: {decline_total:.0f} units ({(decline_total/lifecycle_total)*100:.1f}% of total)")
print(f"Total Lifecycle Sales: {lifecycle_total:.0f} units")
```

## Advanced Topics

### Multiple Promotional Events

```python
from sim_lab.core import BaseSimulation, SimulatorRegistry
import numpy as np
from typing import List, Optional, Dict, Any

@SimulatorRegistry.register("MultiPromoProduct")
class MultiPromoProductSimulation(BaseSimulation):
    """Product simulation with multiple promotional events."""
    
    def __init__(
        self,
        start_demand: float,
        days: int,
        growth_rate: float,
        marketing_impact: float,
        promotions: List[Dict[str, Any]] = None,
        random_seed: Optional[int] = None
    ):
        """Initialize the product simulation with multiple promotions."""
        super().__init__(days=days, random_seed=random_seed)
        self.start_demand = start_demand
        self.growth_rate = growth_rate
        self.marketing_impact = marketing_impact
        self.promotions = promotions or []
    
    def run_simulation(self) -> List[float]:
        """Run the simulation with multiple promotional events."""
        self.reset()
        
        demand = [self.start_demand]
        for day in range(1, self.days):
            previous_demand = demand[-1]
            natural_growth = previous_demand * (1 + self.growth_rate)
            marketing_influence = previous_demand * self.marketing_impact
            
            new_demand = natural_growth + marketing_influence
            
            # Check for promotions on this day
            for promo in self.promotions:
                if day == promo.get("day", 0):
                    new_demand *= (1 + promo.get("effectiveness", 0))
            
            demand.append(new_demand)
            
        return demand

# Create and run the simulation with multiple promotions
multi_promo = MultiPromoProductSimulation(
    start_demand=100,
    days=365,
    growth_rate=0.002,
    marketing_impact=0.001,
    promotions=[
        {"day": 30, "effectiveness": 0.3, "name": "Launch Event"},
        {"day": 90, "effectiveness": 0.2, "name": "Summer Sale"},
        {"day": 180, "effectiveness": 0.5, "name": "Black Friday"},
        {"day": 270, "effectiveness": 0.25, "name": "Spring Campaign"}
    ],
    random_seed=42
)

demand = multi_promo.run_simulation()

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(demand)

# Mark the promotions
colors = ['g', 'b', 'r', 'purple']
for i, promo in enumerate(multi_promo.promotions):
    day = promo.get("day", 0)
    name = promo.get("name", f"Promotion {i+1}")
    effectiveness = promo.get("effectiveness", 0)
    plt.axvline(x=day, color=colors[i % len(colors)], linestyle='--')
    plt.text(day+5, demand[day], f"{name} (+{effectiveness*100:.0f}%)", 
             color=colors[i % len(colors)], fontweight='bold')

plt.title('Product Demand with Multiple Promotional Events')
plt.xlabel('Days')
plt.ylabel('Daily Demand (units)')
plt.grid(True)
plt.show()
```

### Competitor Impact Model

```python
from sim_lab.core import BaseSimulation, SimulatorRegistry
import numpy as np
from typing import List, Optional

@SimulatorRegistry.register("CompetitiveProductMarket")
class CompetitiveProductMarketSimulation(BaseSimulation):
    """Product simulation with competitive market effects."""
    
    def __init__(
        self,
        product_a_start_demand: float,
        product_b_start_demand: float,
        days: int,
        product_a_growth: float,
        product_a_marketing: float,
        product_b_growth: float,
        product_b_marketing: float,
        competition_factor: float,
        market_size_limit: float,
        random_seed: Optional[int] = None
    ):
        """Initialize the competitive market simulation."""
        super().__init__(days=days, random_seed=random_seed)
        self.product_a_start = product_a_start_demand
        self.product_b_start = product_b_start_demand
        self.product_a_growth = product_a_growth
        self.product_a_marketing = product_a_marketing
        self.product_b_growth = product_b_growth
        self.product_b_marketing = product_b_marketing
        self.competition_factor = competition_factor
        self.market_size_limit = market_size_limit
    
    def run_simulation(self) -> Dict[str, List[float]]:
        """Run the competitive market simulation."""
        self.reset()
        
        # Initialize demand arrays
        demand_a = [self.product_a_start]
        demand_b = [self.product_b_start]
        
        for day in range(1, self.days):
            # Get previous demand
            prev_a = demand_a[-1]
            prev_b = demand_b[-1]
            total_market = prev_a + prev_b
            
            # Apply market size limit (logistic growth constraint)
            market_saturation = total_market / self.market_size_limit
            saturation_factor = max(0, 1 - market_saturation)
            
            # Calculate natural growth and marketing
            growth_a = prev_a * self.product_a_growth * saturation_factor
            marketing_a = prev_a * self.product_a_marketing
            
            growth_b = prev_b * self.product_b_growth * saturation_factor
            marketing_b = prev_b * self.product_b_marketing
            
            # Competition effect (products take share from each other)
            competition_a = -prev_a * self.competition_factor * (prev_b / total_market)
            competition_b = -prev_b * self.competition_factor * (prev_a / total_market)
            
            # Calculate new demand
            new_a = prev_a + growth_a + marketing_a + competition_a
            new_b = prev_b + growth_b + marketing_b + competition_b
            
            # Ensure non-negative values
            demand_a.append(max(0, new_a))
            demand_b.append(max(0, new_b))
            
        return {"product_a": demand_a, "product_b": demand_b}

# Create and run the competitive market simulation
market = CompetitiveProductMarketSimulation(
    product_a_start_demand=200,  # Established product
    product_b_start_demand=50,   # New entrant
    days=730,                    # Two years
    product_a_growth=0.001,      # Slower growth (mature product)
    product_a_marketing=0.001,   # Standard marketing
    product_b_growth=0.008,      # Faster growth (new product)
    product_b_marketing=0.005,   # Aggressive marketing
    competition_factor=0.002,    # Competition intensity
    market_size_limit=10000,     # Total market size
    random_seed=42
)

results = market.run_simulation()

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(results["product_a"], 'b-', label='Product A (Established)')
plt.plot(results["product_b"], 'r-', label='Product B (Challenger)')
plt.plot(np.array(results["product_a"]) + np.array(results["product_b"]), 
         'g--', label='Total Market')

plt.title('Competitive Product Market Dynamics')
plt.xlabel('Days')
plt.ylabel('Daily Demand (units)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate market share over time
total_market = np.array(results["product_a"]) + np.array(results["product_b"])
share_a = np.array(results["product_a"]) / total_market
share_b = np.array(results["product_b"]) / total_market

plt.figure(figsize=(14, 7))
plt.stackplot(range(market.days), share_a, share_b, 
              labels=['Product A', 'Product B'],
              colors=['blue', 'red'])
plt.title('Market Share Evolution')
plt.xlabel('Days')
plt.ylabel('Market Share (%)')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
```

## API Reference

### ProductPopularitySimulation

```python
ProductPopularitySimulation(
    start_demand: float,
    days: int,
    growth_rate: float,
    marketing_impact: float,
    promotion_day: Optional[int] = None,
    promotion_effectiveness: float = 0,
    random_seed: Optional[int] = None
)
```

#### Methods

- **run_simulation()**: Run the simulation and return a list of demand values over time.
- **reset()**: Reset the simulation to its initial state.

## Further Reading

For more information about product lifecycle and demand modeling:

- [Product Lifecycle](https://en.wikipedia.org/wiki/Product_lifecycle)
- [Diffusion of Innovations](https://en.wikipedia.org/wiki/Diffusion_of_innovations)
- [Bass Diffusion Model](https://en.wikipedia.org/wiki/Bass_diffusion_model)
- [Marketing Mix Modeling](https://en.wikipedia.org/wiki/Marketing_mix_modeling)