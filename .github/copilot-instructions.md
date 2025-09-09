# Copilot Instructions for Pyro-Meets-SBI Tutorial

You are an agent working on the **Pyro-Meets-SBI** tutorial materials for EuroSciPy 2025. This project demonstrates hierarchical Bayesian inference by combining Pyro's probabilistic programming with SBI's simulation-based inference capabilities.

## Core Project Context

- **Purpose**: Teaching hierarchical Bayesian inference using the cookie factory example
- **Key Concepts**: Pooled, unpooled, and hierarchical models with Pyro and SBI
- **Target Audience**: Scientists and researchers with basic Python knowledge
- **Conference**: EuroSciPy 2025, Kraków, Poland
- **Duration**: 20-minute talk + 90-minute hands-on tutorial

## Project Structure

```
pyro-meets-sbi/
├── notebooks/
│   ├── 01_pyro_cookie_example.ipynb      # Standard Pyro hierarchical model
│   ├── 02_pyro_sbi_cookie_example.ipynb  # Pyro with SBI neural likelihood
│   └── 03_comparison.ipynb               # Side-by-side comparison
├── src/
│   ├── cookie_simulator.py               # Cookie chip count simulator
│   ├── pyro_models.py                    # Pooled, unpooled, hierarchical models
│   └── sbi_wrapper.py                    # ConditionalDensityEstimator wrapper
├── data/
│   └── cookie_chips_data.csv             # 5 locations × 30 cookies data
├── figures/
│   └── chips_per_location.png            # Distribution visualization
└── copilot-instructions.md               # This file
```

## The Cookie Factory Example

### Scenario
- 5 cookie factory locations
- 30 cookies sampled per location
- Count chocolate chips in each cookie
- Goal: Estimate chip rate per location while accounting for global patterns

### Three Modeling Approaches

1. **Pooled Model**: All locations share the same rate
2. **Unpooled Model**: Each location is completely independent  
3. **Hierarchical Model**: Locations are different but related (partial pooling)

## Key Implementation Guidelines

### When Creating Pyro Models

```python
# Always use plate notation for conditional independence
with pyro.plate("location", n_locations):
    # Location-specific parameters
    
with pyro.plate("data", n_observations):
    # Observations
```

### When Integrating SBI

```python
# 1. Train neural likelihood estimator
from sbi.inference import NLE
nle = NLE(prior)
estimator = nle.append_simulations(theta, x).train()

# 2. Wrap for Pyro
class ConditionalDensityEstimatorDistribution(pyro.distributions.TorchDistribution):
    def __init__(self, estimator, condition):
        self.estimator = estimator
        self.condition = condition
        
    def log_prob(self, value):
        return self.estimator.log_prob(value, condition=self.condition)
```

## Notebook Development Workflow

### For `01_pyro_cookie_example.ipynb`

1. **Setup Section**
   - Import pyro, torch, matplotlib
   - Set random seeds for reproducibility
   - Load or generate cookie data

2. **Data Exploration**
   - Show per-location statistics
   - Visualize distributions
   - Motivate hierarchical approach

3. **Model Implementation**
   - Implement pooled model
   - Implement unpooled model
   - Implement hierarchical model
   - Use clear variable names (mu, sigma, lam)

4. **Inference**
   - Use NUTS for MCMC
   - Run with appropriate warmup and samples
   - Handle divergences if they occur

5. **Analysis**
   - Show shrinkage effect
   - Compare posterior distributions
   - Posterior predictive checks

### For `02_pyro_sbi_cookie_example.ipynb`

1. **Simulator Definition**
   ```python
   def cookie_simulator(params):
       # Simulate chip counts given rate parameter
       # Must handle batched inputs
       return torch.poisson(params.unsqueeze(-1).expand(-1, 30))
   ```

2. **SBI Training**
   - Generate training data (10,000 simulations)
   - Train NLE with appropriate architecture
   - Validate with simulation-based calibration

3. **Hierarchical Model with SBI**
   - Same hyperprior structure as standard Pyro
   - Replace Poisson likelihood with neural likelihood
   - Demonstrate identical inference results

4. **Comparison**
   - Show that results match standard Pyro
   - Discuss when SBI is necessary
   - Performance considerations

## Code Style Requirements

### Python Style
- **Docstrings**: Google style with full parameter documentation
- **Type hints**: Always use, especially for function signatures
- **Line length**: Maximum 88 characters
- **Imports**: Group by standard library, third-party, local

### Notebook Style
- **Markdown headers**: Clear section divisions
- **Cell outputs**: Keep important visualizations
- **Code comments**: Explain non-obvious steps
- **Magic commands**: Use `%matplotlib inline` at start

## Testing and Validation

### For Standard Pyro Models
```python
# Always check MCMC diagnostics
assert mcmc.diagnostics()["r_hat"].max() < 1.01
assert mcmc.diagnostics()["ess"].min() > 400
```

### For SBI Integration
```python
# Validate neural likelihood training
assert estimator.train_losses[-1] < initial_loss * 0.5

# Check coverage with SBC
coverage = run_sbc(estimator, prior, n_simulations=100)
assert 0.4 < coverage < 0.6  # Should be ~0.5
```

## Common Tasks and Solutions

### "Make the models more accessible to beginners"
1. Add more explanatory markdown cells
2. Use intuitive variable names
3. Include intermediate print statements
4. Visualize each step
5. Provide interpretation of results

### "Optimize notebook performance"
1. Reduce MCMC samples for quick runs
2. Cache trained SBI models
3. Use smaller batch sizes if memory limited
4. Consider GPU acceleration for SBI

### "Add diagnostic plots"
1. Trace plots for MCMC chains
2. Posterior predictive distributions
3. Prior predictive checks
4. Shrinkage visualization
5. SBC rank histograms

## Visualization Guidelines

### Color Scheme
```python
COLORS = {
    'hierarchical': '#E69F00',  # Orange
    'simulation': '#009E73',     # Green  
    'bayesian': '#0072B2',       # Blue
    'pyro': '#CC79A7'           # Pink
}
```

### Standard Plots
- Histograms with KDE overlays
- Credible intervals as shaded regions
- Location comparison as grouped bars
- Use colorblind-friendly palettes

## Error Handling

### Common Issues
1. **Divergences in NUTS**: Increase target_accept_prob
2. **SBI training instability**: Reduce learning rate
3. **Memory errors**: Reduce batch size
4. **Slow inference**: Use vectorized operations

## Documentation Requirements

### Each Notebook Must Include
1. **Title and purpose** (first cell)
2. **Learning objectives** (what readers will understand)
3. **Prerequisites** (assumed knowledge)
4. **Data description** (source and structure)
5. **Model equations** (mathematical formulation)
6. **Results interpretation** (what the output means)
7. **Key takeaways** (summary points)

## Resources to Reference

### Documentation
- **Pyro**: https://docs.pyro.ai/
- **SBI**: https://sbi-dev.github.io/sbi/
- **PyTorch**: https://pytorch.org/docs/

### Key Papers
- Cranmer et al. (2020): "The frontier of simulation-based inference"
- Gelman et al. (2013): "Bayesian Data Analysis" (for hierarchical models)
- Orduz blog: Cookie factory example inspiration

## Validation Checklist

Before considering a notebook complete:

- [ ] Runs end-to-end without errors
- [ ] Clear narrative flow from problem to solution
- [ ] All code cells have explanatory markdown
- [ ] Visualizations are clear and labeled
- [ ] Results match expected behavior
- [ ] Inference diagnostics are satisfactory
- [ ] Key concepts are highlighted
- [ ] Comparison between approaches is clear
- [ ] Accessibility features included (alt text, colorblind-friendly)
- [ ] Tested on CPU (no GPU requirement)

## Communication Style

When developing these notebooks:
- Use clear, educational language
- Avoid jargon without explanation
- Build concepts progressively
- Relate to real-world intuition
- Encourage experimentation
- Acknowledge common confusion points

## Remember

- You're creating educational materials for EuroSciPy 2025
- The cookie example should be simple but illustrative
- Show the power of combining Pyro and SBI
- Make the content accessible to non-experts
- Demonstrate when SBI is necessary vs. when standard Pyro suffices
- Always test that notebooks run completely before finalizing

## Final Notes

These notebooks will be used in:
1. A 20-minute conference talk (high-level overview)
2. A 90-minute tutorial (hands-on coding)
3. Post-conference self-study

Ensure they work for all three contexts by providing both quick examples and detailed explorations.
