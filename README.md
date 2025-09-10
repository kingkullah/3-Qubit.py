# 3-Qubit Quantum Circuit Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A Python implementation for simulating 3-qubit quantum circuits, demonstrating quantum gates, superposition, and entanglement. This project provides both computational simulation and interactive visualization of quantum states.

### Features
- 3-qubit quantum state simulation
- Support for common quantum gates (Hadamard, CNOT, Pauli gates)
- Interactive visualization capabilities
- Measurement probability calculations
- Extensible framework for quantum circuit experiments

## Installation (Ubuntu/WSL2)

### Prerequisites
```bash
# Update package list
sudo apt update

# Install Python 3.10+ and pip
sudo apt install -y python3 python3-pip python3-venv

# Install git if not already installed
sudo apt install -y git

git clone https://github.com/kingkullah/3-Qubit.py.git
cd 3-Qubit.py
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
from src.qubits.three_qubit import QuantumCircuitSimulator

# Initialize simulator
simulator = QuantumCircuitSimulator()

# Create Hadamard gate
h_gate = simulator.create_hadamard()
print(f"Hadamard gate: {h_gate}")

# Apply CNOT gate
cnot = simulator.apply_cnot_gate(0, 1, 3)
print(f"CNOT gate shape: {cnot.shape}")
# Run basic 3-qubit simulation
python src/qubits/three_qubit.py
# Install in development mode first
pip install -e .

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html
3-Qubit-Simulation/
├── src/qubits/           # Main source code
├── notebooks/            # Jupyter examples  
├── tests/               # Unit tests
├── docs/                # Documentation
├── .github/workflows/   # CI/CD pipelines
└── requirements.txt     # Dependencies
@misc{3qubit_simulation,
  title={3-Qubit Quantum Circuit Simulation},
  author={kingkullah},
  year={2024},
  url={https://github.com/kingkullah/3-Qubit.py}
}
**What this does:** Creates a comprehensive README with installation and usage instructions
**Safety:** Safe - just replaces the existing 40-byte README with a detailed one

Please run these two commands and let me know if you see any errors!
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 kingkullah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
