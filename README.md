# ChemNODE-Ammonia: Fast Surrogate Modeling for Stiff Ammonia Combustion Kinetics

**Author:** Aditya Prasad  
**Course:** ME 691: SciML for Thermo-fluids (IIT Gandhinagar)

## Project Overview
Simulating ammonia combustion is computationally expensive due to the **numerical stiffness** of detailed chemical mechanisms (e.g., Stagni 2023, 31 species). This project implements a **Neural Ordinary Differential Equation (Neural ODE)** framework to create a fast surrogate model.

The model learns the continuous-time chemical dynamics ($d\mathbf{y}/dt$) and integrates them using a JIT-compiled explicit solver, achieving a significant speed-up of around 85x over traditional stiff solvers (CVODE/BDF) while maintaining high accuracy.

## Features
* **Mechanism:** Stagni et al. (2023) Ammonia/Hydrogen mechanism.
* **Architecture:** 4-layer MLP (128 neurons) wrapped in a Neural ODE.
* **Physics-Aware:** Trains on reaction trajectories to minimize integration error.
* **Performance:** Achieves a **[85]x speed-up** using a custom JIT-compiled RK4 solver on GPU.

## Repository Structure
* `notebooks/`: Jupyter notebooks for the 4-phase pipeline (Generation, Preprocessing, Training, Eval).
* `data/`: Contains the chemical mechanism and normalization parameters.
* `model weights/`: Trained PyTorch model weights.

## Usage

### 1. Installation
```bash
pip install -r requirements.txt
