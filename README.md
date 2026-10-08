# Logarithmic Quantum Computer Vision: Amplitude Encoded QCNN

A hardware-accelerated **Hybrid Quantum-Classical Convolutional Neural Network (QCNN)** implementing high-density **Logarithmic Amplitude Encoding** via PennyLane and PyTorch. This architecture acts as Phase 2 of a comparative quantum machine learning portfolio, designed to be directly benchmarked against a 1-to-1 linear Angle Encoding pipeline.

---

## 🔬 Core Paradigm Shift: Angle vs. Amplitude Encoding

Unlike linear Angle Encoding which assigns a single feature to a single qubit (O(N) scaling), **Amplitude Encoding** maps features directly into the probability amplitudes of a quantum state superposition. 

Because an N-qubit system inherently possesses \(2^N\) complex amplitudes, feature storage capacity scales **logarithmically** (\(O(\log_2 M)\) features for N qubits).

### Architectural Upgrades in This Repository:
* **High-Density Patching:** A 4-qubit register can now instantly ingest a full **4x4 spatial patch (16 pixel features)** in a single quantum operation, completely replacing the tight 2x2 bottleneck of the previous pipeline.
* **Global Embedding Scalability:** If scaled up to an 8-qubit register (2⁸ = 256), the network can embed a complete **14x14 normalized image (196 features)** directly into the state vector, removing sliding-window constraints entirely.


---

## 🧮 Mathematical Formulation

### 1. State Superposition Injection
Given an input vector 

```math
x = [x_0, x_1, \dots, x_{M-1}]^T \in \mathbb{R}^M where M \le 2^N 
```
the data is normalized such that 
```math
\vert{}\x\vert{}_2 = 1
```
The state preparation unitary maps these normalized values directly to computational basis amplitudes:

```math

|\psi(x)\rangle = \sum_{i=0}^{M-1} x_i |i\rangle
```

### 2. PennyLane Implementation
Instead of relying on a sequential loop of independent parametric single-qubit rotations, this framework leverages high-density state vector preparation:
```python
qml.AmplitudeEmbedding(features=inputs, wires=range(4), normalize=True)
```

---



## 📁 Repository Structure Blueprint

```text
logarithmic-qcnn/
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── dataset.py          # 14x14 pre-processing with flat 16-feature vector outputs
    ├── quantum_filter.py   # 4-qubit qml.AmplitudeEmbedding VQC
    ├── model.py            # Post-processing PyTorch CNN for 4x4 spatial strides
    └── train.py            # Unified PyTorch-MPS orchestration engine
```

---

## 🚀 Getting Started

1. Initialize your isolated project virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pennylane "autoray<0.8.0"
```
