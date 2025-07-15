# ChemKinetics-NODE

Neural ODE Framework for Efficient and Accurate Chemical Kinetics Simulation in Methane Combustion

## Overview

This project implements a Neural Ordinary Differential Equations (NODE) approach to model and predict chemical kinetics in methane combustion reactions. By leveraging deep learning techniques, the model learns the underlying dynamics of chemical species concentrations and temperature evolution over time, achieving high accuracy while significantly reducing computational costs compared to traditional solvers.

## Key Features

- **Neural ODE Architecture**: Uses a deep learning model to learn chemical reaction dynamics
- **Time-Series Prediction**: Accurately forecasts the evolution of chemical species concentrations
- **Multi-Scale Modeling**: Captures both fast local dynamics and slower global changes
- **Computational Efficiency**: Achieves significant speedup compared to traditional chemical kinetics solvers
- **GRI 3.0 Mechanism**: Based on the methane combustion mechanism with comprehensive evaluation

## Results

The model achieves excellent accuracy in predicting species concentrations and temperature:

| Species | RMSE (Test) | MAE (Test) | NMAE (Test) |
|---------|-------------|------------|-------------|
| CH4     | 5.5760e-04  | 2.3858e-04 | 8.4530e-01 |
| O2      | 3.8184e-04  | 2.0640e-04 | 9.5330e-01 |
| H2O     | 2.1349e-03  | 1.1914e-03 | 7.8603e-01 |
| CO2     | 2.6732e-02  | 2.2125e-02 | 1.3101e-01 |
| T       | 2.5455e+02  | 2.1017e+02 | 1.1879e-01 |

## Tech Stack

- PyTorch
- torchdiffeq
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Cantera (optional, for comparison)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ChemKinetics-NODE.git
cd ChemKinetics-NODE

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Example code to load and run the model
import torch
from model import CombustionODE
from torchdiffeq import odeint

# Load the trained model
model = CombustionODE(dim=9)  # Number of species + temperature
model.load_state_dict(torch.load('best.pth'))

# Initial conditions
y0 = torch.tensor([...])  # Initial concentrations and temperature
t = torch.linspace(0, 1.0, 100)  # Time points

# Run prediction
with torch.no_grad():
    pred = odeint(model, y0, t)

# Process results
# ...
```

## Dataset

The model is trained on time-series data of methane combustion reactions under various initial conditions. The dataset includes:

- Training data: Multiple trajectories with different initial conditions
- Testing data: Trajectories with initial conditions outside the training set
- Features: Concentrations of key species (CH4, O2, H2O, CO2, etc.) and temperature

## Model Architecture

The model uses a Neural ODE approach with:
- Feedforward neural network to learn the dynamics
- Multiple hidden layers with ReLU activations
- Dropout for regularization
- Adjoint method for memory-efficient backpropagation

## Citation

If you use this code in your research, please cite:

```
@article{chemnode2021,
  title={ChemNODE: A Neural Ordinary Differential Equations Framework for Efficient Chemical Kinetic Solvers},
  author={Owoyele, Opeoluwa and Pal, Pinaki},
  journal={arXiv preprint arXiv:2101.04749},
  year={2021}
}
```

## License

MIT

## Acknowledgements

This project is inspired by the ChemNode paper by Owoyele and Pal, which introduced the Neural ODE approach for chemical kinetics simulation. 