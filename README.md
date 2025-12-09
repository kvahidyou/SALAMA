# SALAMA1D

## Overview
**SALAMA 1D** is a deep neural network model designed to infer the probability of thunderstorm occurrence from the vertical profiles of ten atmospheric fields from numerical weather prediction (NWP). It has been trained using
- operational forecasts of the ICON-D2-EPS weather model, provided by [Deutscher Wetterdienst](https://www.dwd.de/)
- lightning observations of the LINET lightning detection network, provided by [nowcast GmbH](https://www.nowcast.de/)

The scientific background is detailed in the following paper:
K. Vahid Yousefnia et al. (2025): Inferring Thunderstorm Occurrence from Vertical Profiles of Convection-Permitting Simulations: Physical Insights from a Physical Deep Learning Model. _Artif. Intell. Earth Syst._, 4, 240096, doi: [10.1175/AIES-D-24-0096.1](https://doi.org/10.1175/AIES-D-24-0096.1)

The datasets used for training, validation, and testing are available from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13981208.svg)](https://doi.org/10.5281/zenodo.13981208).

This repository provides the relevant code to use SALAMA 1D. In addition, we provide the code for SALAMA 0D, which uses single-level fields instead of vertical profiles is used as a benchmark in the above-mentioned paper.

Fun fact: The acronym stands for **S**ignature-based **A**pproach of identifying **L**ightning **A**ctivity using **Ma**chine learning and means "lightning" in Finnish.



## Project Structure

```plaintext
SALAMA1D/
│
├── README.md                # Project documentation (this file)
├── examples/                # Example scripts and usage demonstrations
├── models/                  # Directory containing model architectures and trained models
│   ├── SALAMA1D/            # Directory containing the SALAMA 1D model and architecture
│   └── SALAMA0D/            # Directory containing the SALAMA 0D model and architecture
└── LICENSE                  # Open-source License File
└── CHANGELOG                # changelog of software updates
```

## Getting Started

### Prerequisites

To use SALAMA 1D, you will need the following:

- **Python 3.8+**
- **[PyTorch](https://pytorch.org/) 1.12+** _(BSD-3 License)_
- **[NumPy](https://numpy.org/) <2** _(modified BSD License)_

You can install the required packages using pip:

```bash
pip install torch numpy<2
```

### Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kvahidyou/SALAMA.git
   cd SALAMA
   ```
2. **Run the example script:**
The example script located in the directory examples/ demonstrates how to use the SALAMA 1D model to compute the probability of thunderstorm occurrence.
```bash
python examples/example_SALAMA1D.py
```

## License
Please see the file LICENSE.md for further information about how the content is licensed.

## Citation
If you use SALAMA 1D in your research, please refer to the specific version on Zenodo [![DOI](https://zenodo.org/badge/891436828.svg)](https://doi.org/10.5281/zenodo.14212888), and cite the following paper:

K. Vahid Yousefnia et al. (2025): Inferring Thunderstorm Occurrence from Vertical Profiles of Convection-Permitting Simulations: Physical Insights from a Physical Deep Learning Model. _Artif. Intell. Earth Syst._, 4, 240096, doi: [10.1175/AIES-D-24-0096.1](https://doi.org/10.1175/AIES-D-24-0096.1)
