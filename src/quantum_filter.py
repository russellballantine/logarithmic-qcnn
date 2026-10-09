# src/quantum_filter.py
import pennylane as qml
import torch

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch")
def quantum_conv_circuit(inputs, weights):
    """
    4-qubit Quantum Convolutional Filter utilizing Amplitude Embedding.
    """
    # Step 1: Inject 16 classical features using the explicitly named 'inputs' parameter
    qml.AmplitudeEmbedding(inputs, wires=range(4), normalize=False)
    
    # Step 2: Variational Ansatz Layer 1 (Rotations)
    for i in range(4):
        qml.RY(weights[i], wires=i)
        qml.RZ(weights[i + 4], wires=i)
        
    # Step 3: Entangling Layer (Ring Entanglement)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    
    # Step 4: Variational Ansatz Layer 2
    for i in range(4):
        qml.RY(weights[i + 8], wires=i)
        
    # Step 5: Quantum Measurement Strategy
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]

