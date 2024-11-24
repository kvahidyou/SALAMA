# SALAMA1D

## Overview
**SALAMA 1D** is a deep neural network model designed to infer the probability of thunderstorm occurrence from the vertical profiles of ten atmospheric fields from numerical weather prediction (NWP). It has been trained using
- operational forecasts of the ICON-D2-EPS weather model, provided by [Deutscher Wetterdienst](https://www.dwd.de/)
- lightning observations of the LINET lightning detection network, provided by [nowcast GmbH](https://www.nowcast.de/)

The scientific background is detailed in the following paper:
K. Vahid Yousefnia et al.: _Inferring Thunderstorm Occurrence from Vertical Profiles of Convection-Permitting Simulations: Physical Insights from a Physical Deep Learning Model_, 2024. (Submitted to _Artificial Intelligence for the Earth Systems_, preprint available at: http://arxiv.org/abs/2409.20087)

This repository provides the relevant code to use SALAMA 1D.

Fun fact: The acronym stands for **S**ignature-based **A**pproach of identifying **L**ightning **A**ctivity using **Ma**chine learning and means "lightning" in Finnish.



## Project Structure

```plaintext
SALAMA1D/
│
├── README.md                # Project documentation (this file)
├── examples/                # Example scripts and usage demonstrations
│   └── example_script.py    # Script demonstrating how to use the SALAMA 1D model
├── models/                  # Directory containing model architectures and trained models
│   └── SALAMA1D_22/         # Directory containing the SALAMA 1D model and architecture
│       ├── architecture.py  # Script defining the deep neural network architecture
│       ├── model.py         # Script defining the SALAMA 1D model class
│       ├── state_dict.pth   # Trained model weights
│       ├── scaling_mean.pt  # Mean values for input normalization
│       └── scaling_std.pt   # Standard deviation values for input normalization
└── LICENSE                  # Open Source License File
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
The example script located in the examples/directory demonstrates how to use the SALAMA 1D model to compute the probability of thunderstorm occurrence.
```bash
python examples/example_script.py
```

## License
Please see the file LICENSE.md for further information about how the content is licensed.

## Citation
If you use SALAMA 1D in your research, please refer to the specific version on Zenodo [![DOI](https://zenodo.org/badge/891436828.svg)](https://doi.org/10.5281/zenodo.14212888), and cite the following paper:

K. Vahid Yousefnia et al.: _Inferring Thunderstorm Occurrence from Vertical Profiles of Convection-Permitting Simulations: Physical Insights from a Physical Deep Learning Model_, 2024. (Submitted to _Artificial Intelligence for the Earth Systems_, preprint available at: http://arxiv.org/abs/2409.20087)
