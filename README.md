# Pyro Meets SBI: Hierarchical Simulation-Based Bayesian Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janfb/pyro-meets-sbi)

## EuroSciPy 2025 Talk Materials

Welcome! This repository contains the slides and supporting materials for the talk "Pyro
Meets SBI: Unlocking Hierarchical Bayesian Inference for Complex Simulators" presented
at EuroSciPy 2025 in Kraków, Poland.

📍 **Talk Details**: [EuroSciPy 2025 - Pyro Meets SBI](https://euroscipy.org/talks/KCYYTF/)  
🎯 **Tutorial**: See also the companion [SBI Tutorial](https://github.com/janfb/euroscipy-2025-sbi-tutorial)

## Quick Links

- 🔧 SBI Package: [github.com/sbi-dev/sbi](https://github.com/sbi-dev/sbi)
- 🔥 Pyro: [pyro.ai](https://pyro.ai)
- 🔄 Pyro-SBI Integration [pull request](https://github.com/sbi-dev/sbi/pull/1491) by [Seth Axen](https://sethaxen.com/)

## Contents

### 📊 Slides

- `slides/` folder with markdown slides and image files
- `src/` folder with jupyter notebooks with code examples

## Talk Abstract

Complex simulators are ubiquitous in science—from neural circuits to climate models—but
often lack tractable likelihood functions. This talk demonstrates how to combine Pyro's
elegant probabilistic programming with Simulation-Based Inference (SBI) to perform
hierarchical Bayesian inference on such models.

### Key Topics Covered

- **Hierarchical Modeling**: Understanding pooled, unpooled, and hierarchical approaches
- **Simulation-Based Inference**: Neural approximation of likelihoods (NPE, NLE, NRE)
- **Practical Integration**: Wrapping SBI estimators as Pyro distributions

### Learning Outcomes

After this talk, you will understand:

1. When and why hierarchical models are beneficial
2. How SBI enables inference for complex simulators
3. How to combine Pyro and SBI in practice

## Installation

The easiest way is with uv (fast Python package manager and envs).

1. Install uv (macOS)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Ensure uv is on PATH (new shells will pick this up)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
uv --version
```

2. Create and activate a virtual environment

```bash
cd pyro-meets-sbi
uv venv .venv -p 3.11
source .venv/bin/activate
```

3. Install dependencies

```bash
uv sync
```

4. (Optional) Register a Jupyter kernel

```bash
python -m ipykernel install --user --name=pyro-meets-sbi
```

## Launching the Notebooks

```bash
jupyter notebook
```

Open:

- `src/01_pyro_cookie_example.ipynb`
- `src/02_pyro-sbi_cookie_example.ipynb`

### Open these notebooks in Colab

[![Open 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janfb/pyro-meets-sbi/blob/master/src/01_pyro_cookie_example.ipynb)
[![Open 02 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janfb/pyro-meets-sbi/blob/master/src/02_pyro-sbi_cookie_example.ipynb)

## Resources

### Papers

- **SBI review paper**: Cranmer et al. (2020) - [The frontier of simulation-based inference](https://www.pnas.org/doi/10.1073/pnas.1912789117)
- **SBI software paper**: Boelts, Deistler et al. (2024) - [sbi reloaded: a toolkit for simulation-based inference workflows](https://joss.theoj.org/papers/10.21105/joss.07754)
- **SBI tutorial paper**: Deistler, Boelts et al. (2025) - [SBI: A practical guide](https://arxiv.org/abs/2508.12939)

### Documentation

- **SBI Documentation**: [sbi.readthedocs.io/en/latest/](https://sbi.readthedocs.io/en/latest/)
- **Pyro Documentation**: [docs.pyro.ai](http://docs.pyro.ai/)
- **EuroSciPy SBI Tutorial**: [Link to GitHub](https://github.com/janfb/euroscipy-2025-sbi-tutorial)

## Acknowledgments

This work has been made possible through the support and contributions of many:

### Communities

- **SBI Community** - For developing and maintaining the `sbi` package, especially Seth Axen for implementing the Pyro wrapper during the SBI Hackathon 2024
  - Special acknowledgment to **Seth Axen** who implemented the wrapper from `sbi` to `pyro` ([sbi-dev/sbi#1491](https://github.com/sbi-dev/sbi/pull/1491))
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

These materials are released under the Apache 2.0 License.

---
