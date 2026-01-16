# Spatial Locking of Chimera States

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

<!-- TOC -->

- [Spatial Locking of Chimera States](#spatial-locking-of-chimera-states)
    - [About](#about)
    - [Paper](#paper)
        - [Abstract](#abstract)
    - [Installation](#installation)
        - [Prerequisites](#prerequisites)
        - [Steps](#steps)
    - [Usage](#usage)
    - [Notebooks Overview](#notebooks-overview)
    - [Pre-commit Hooks](#pre-commit-hooks)
        - [Setting Up Pre-commit](#setting-up-pre-commit)
    - [Citation](#citation)

<!-- /TOC -->

## About

This repository contains the numerical simulation code and analysis tools for studying **spatial locking of chimera states** in nonlocally coupled oscillator systems with frequency heterogeneity.

Chimera states are fascinating self-organized phenomena where synchronized (coherent) and desynchronized (incoherent) oscillator populations coexist in spatially extended systems. This work investigates how frequency variations across the system influence the spatial organization of these chimera states, demonstrating that coherent and incoherent domains align with lower and higher frequency regions respectively.

## Paper

**Spatial locking of chimera states to frequency heterogeneity in nonlocally coupled oscillators**

*Petar Mircheski and Hiroya Nakao*

- **Published in:** [Chaos (AIP Publishing)](https://pubs.aip.org/aip/cha/article/35/6/063105/3348097/Spatial-locking-of-chimera-states-to-frequency)
- **arXiv:** [https://arxiv.org/abs/2502.02881](https://arxiv.org/abs/2502.02881)

### Abstract

Coupled nonlinear oscillators are ubiquitous in chemical, mechanical, electrical, optical, and biological systems and often play important functional roles in individual systems. 
Depending on the type of coupling, the oscillators can exhibit a variety of intriguing collective dynamics.
Among them, the chimera state that arises under nonlocal coupling has attracted much attention as a peculiar phenomenon characterized by self-organized coexistence of coherent and incoherent oscillator domains.
In this study, we analyze the influence of weak frequency heterogeneity on chimera states. 
We observe that the coherent domain tends to shift away from regions of higher frequency and align with regions of lower frequency, resulting in self-adaptive spatial locking.
Our findings provide insight into the role of frequency heterogeneity in the formation and dynamics of chimera states with translational symmetry

## Installation

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) (dependency management)

### Steps

1. Clone the repository:
    ```sh
    git clone git@github.com:PetarMirceski/spatial_locking.git
    cd spatial_locking
    ```

2. Install dependencies using Poetry:
    ```sh
    poetry install
    poetry shell
    ```

## Usage

After activating the poetry shell, you can run the Jupyter notebooks to reproduce the paper's results:

```sh
poetry shell
code . 
```

or 

```sh
poetry shell
jupyter lab 
```

## Notebooks Overview

| Notebook | Description |
|----------|-------------|
| `cosine.ipynb` | Main analysis demonstrating spatial locking with cosine forcing h(x) = cos(x + beta) |
| `dirac.ipynb` | Analysis with localized Dirac delta forcing |
| `chimera_homogenous.ipynb` | Study of chimera states in homogeneous systems |
| `position_plot.ipynb` | Visualization of spatial profiles R(x), Phi(x) |

## Pre-commit Hooks

This project uses pre-commit to maintain code quality and consistency.

### Setting Up Pre-commit

Install pre-commit hooks:
```sh
pre-commit install
```

Run pre-commit hooks manually on all files:
```sh
pre-commit run --all-files
```

Or without activating the shell:
```sh
poetry run pre-commit run --all-files
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mircheski2025spatial,
  title={Spatial locking of chimera states to frequency heterogeneity in nonlocally coupled oscillators},
  author={Mircheski, Petar and Nakao, Hiroya},
  journal={Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume={35},
  number={6},
  year={2025},
  publisher={AIP Publishing}
}
```
