# Pyro Meets SBI: Hierarchical Simulation-Based Bayesian Inference

**EuroSciPy 2025 Talk Materials**

Welcome! This repository contains the slides and supporting materials for the talk "Pyro Meets SBI: Unlocking Hierarchical Bayesian Inference for Complex Simulators" presented at EuroSciPy 2025 in Kraków, Poland.

📍 **Talk Details**: [EuroSciPy 2025 - Pyro Meets SBI](https://euroscipy.org/talks/KCYYTF/)  
🎯 **Tutorial**: See also the companion [SBI Tutorial](https://github.com/janfb/euroscipy-2025-sbi-tutorial) (90-minute hands-on session)

## Quick Links

- 🔧 **SBI Package**: [github.com/sbi-dev/sbi](https://github.com/sbi-dev/sbi)
- 🔥 **Pyro**: [pyro.ai](https://pyro.ai)
- 🔄 **Pyro-SBI Integration PR**: [sbi#1491](https://github.com/sbi-dev/sbi/pull/1491)

## Contents

### 📊 Slides

- `slides/` folder with markdown slides and image files
- `src/` folder with jupyter notebooks with code examples

## Viewing the Slides

### Option 1: VS Code with MARP Extension (Recommended)

1. Install [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
2. Open `slides.md` in VS Code
3. Preview with `Ctrl+K V` (Windows/Linux) or `Cmd+K V` (Mac)
4. Present: Command Palette → "Marp: Export Slide Deck..."

### Option 2: MARP CLI
```bash
# Install MARP CLI
npm install -g @marp-team/marp-cli

# Generate HTML presentation
marp slides.md -o slides.html

# Generate PDF handout
marp slides.md -o slides.pdf

# Start presentation server (with hot reload)
marp -s slides.md
```

### Option 3: Online Viewer

Upload `slides.md` to [Marp Web](https://web.marp.app/)

## Talk Abstract

Complex simulators are ubiquitous in science—from neural circuits to climate models—but often lack tractable likelihood functions. This talk demonstrates how to combine Pyro's elegant probabilistic programming with Simulation-Based Inference (SBI) to perform hierarchical Bayesian inference on such models.

### Key Topics Covered

- **Hierarchical Modeling**: Understanding pooled, unpooled, and hierarchical approaches
- **Simulation-Based Inference**: Neural approximation of likelihoods (NPE, NLE, NRE)
- **Practical Integration**: Wrapping SBI estimators as Pyro distributions

### Learning Outcomes
After this talk, you will understand:
1. When and why hierarchical models are beneficial
2. How SBI enables inference for complex simulators
3. How to combine Pyro and SBI in practice
4. Best practices for diagnostic checks

## Installation

TBD.

## Resources

### Papers

- **SBI Review**: Cranmer et al. (2020) - [The frontier of simulation-based inference](https://www.pnas.org/doi/10.1073/pnas.1912789117)
- **SBI Package**: Boelts, Deistler et al. (2024) - [sbi reloaded: a toolkit for simulation-based inference workflows
](https://joss.theoj.org/papers/10.21105/joss.07754)
- **SBI Tutorial paper: A Deistler, Boelts et al. (2025) - [SBI: A practical guide](https://arxiv.org/abs/2508.12939)


### Documentation

- **SBI Documentation**: [sbi.readthedocs.io/en/latest/](https://sbi.readthedocs.io/en/latest/)
- **Pyro Documentation**: [docs.pyro.ai](http://docs.pyro.ai/)
- **EuroSciPy SBI Tutorial**: [Link to GitHub](https://github.com/janfb/euroscipy-2025-sbi-tutorial)


## Acknowledgments

This work has been made possible through the support and contributions of many:

### Communities

- **SBI Community** - For developing and maintaining the `sbi` package, especially Seth Axen for implementing the Pyro wrapper during the SBI Hackathon 2024
- **Pyro Community** - For creating an elegant probabilistic programming framework
- **EuroSciPy 2025 Organizers** - For providing a platform to share this work

### Institutions

- **appliedAI Institute for Europe** - For supporting open-source scientific software development
- **University of Tübingen** - For funding and research support for `sbi`

## Contact

**Jan Teusen** (né Boelts)  
[TransferLab](https://transferlab.ai/), appliedAI Institute for Europe  
🔗 [janfb.github.io](https://janfb.github.io/)

## License

These materials are released under the MIT License. Feel free to use, modify, and distribute with attribution.

---