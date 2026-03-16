# Quantum State Tomography using Machine Learning

Reconstructing quantum states from measurement data using machine learning techniques.

This project simulates quantum measurements in the **Pauli X, Y, and Z bases** and reconstructs the quantum **density matrix** using multiple approaches including neural networks, Bayesian inference, and compressed sensing.

---

# Project Motivation

Quantum state tomography is a fundamental problem in quantum information science.

The goal is to reconstruct the **density matrix** of a quantum system from measurement outcomes.

Traditional tomography methods scale poorly with system size. This project explores how **machine learning techniques** can help reconstruct quantum states efficiently.

---

# Mathematical Background

For a single qubit, the density matrix can be written as

\[
\rho = \frac{1}{2}(I + r_x X + r_y Y + r_z Z)
\]

where

- \(r_x, r_y, r_z\) are the **Bloch vector components**
- \(X, Y, Z\) are **Pauli matrices**

Measurements in the X, Y, Z bases estimate the expectation values:

\[
\langle X \rangle, \langle Y \rangle, \langle Z \rangle
\]

These values determine the Bloch vector and therefore the density matrix.

---

# Methods Implemented

## 1 Neural Network Tomography

A neural network is trained to map measurement statistics to Bloch vector components.

P(+X), P(-X), <X>
P(+Y), P(-Y), <Y>
P(+Z), P(-Z), <Z>


Output:

rx, ry, rz


Architecture:

MLP Regressor
2 Hidden Layers
128 neurons each
ReLU activation


---

## 2 Bayesian Tomography

Bayesian inference estimates the most likely quantum state given measurement outcomes.

Likelihood:

\[
P(data | \rho)
\]

Posterior:

\[
P(\rho | data)
\]

The most probable Bloch vector is obtained from the posterior distribution.

---

## 3 Compressed Sensing Tomography

Quantum states are often low rank.

Compressed sensing reconstructs the density matrix by solving a constrained optimization problem.

Constraints:

- Density matrix is Hermitian
- Positive semidefinite
- Trace = 1

---

# Project Structure
quantum-state-tomography-ml
│
├── README.md
├── requirements.txt
│
├── notebooks
│ └── quantum_state_tomography_project.ipynb
│
├── src
│ ├── quantum_utils.py
│ ├── simulation.py
│ ├── models.py
│ ├── metrics.py
│ ├── visualization.py
│ └── main.py


---

# Example Workflow

1 Generate random quantum state

2 Simulate measurements in

X basis
Y basis
Z basis


3 Train neural network on measurement statistics

4 Reconstruct density matrix

5 Compare reconstructed state with true state using fidelity

---

# Example Measurement Statistics

| Basis | n(+) | n(-) | Estimated Expectation |
|-----|-----|-----|-----|
| X | 882 | 118 | 0.764 |
| Y | 198 | 802 | -0.604 |
| Z | 350 | 650 | -0.300 |

---

# Example Results

Typical reconstruction fidelities:

| Method | Fidelity |
|------|------|
| Neural Network | ~0.995 |
| Bayesian | ~0.993 |
| Compressed Sensing | ~0.996 |

---

# Example Density Matrix

True State
[ 0.65 0.28+0.41i ]
[ 0.28-0.41i 0.35 ]

Predicted State
[ 0.64 0.29+0.40i ]
[ 0.29-0.40i 0.36 ]


---

# Visualization

The project visualizes

- Density matrix heatmaps
- Bloch vectors
- Fidelity comparison
- Measurement statistics


---

# Future Work

This project can be extended to:

### Multi-qubit tomography

State dimension grows as

\[
2^n \times 2^n
\]

### Entangled states

Reconstruct Bell states and GHZ states.

### Experimental data

Use real quantum hardware from

- IBM Quantum
- IonQ
- Rigetti

### Physics-informed neural networks

Combine quantum physics constraints with neural networks.

---

# References

Nielsen & Chuang  
Quantum Computation and Quantum Information

Gross et al.  
Quantum State Tomography via Compressed Sensing

Carleo & Troyer  
Neural Network Quantum States

---

# Author

Vishal Chowdhary

Physics • Machine Learning • Quantum Computing


Input features:

