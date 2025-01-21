import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from typing import List, Tuple

def normalize_features_for_angles(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to map them to angles between 0 and 2π.
    """
    min_vals = np.min(features)
    max_vals = np.max(features)
    
    if max_vals == min_vals:
        return np.zeros_like(features)
        
    normalized = (features - min_vals) / (max_vals - min_vals)
    return normalized * 2 * np.pi

def create_angle_encoding_circuit(data_point: np.ndarray, num_qubits: int) -> Tuple[QuantumCircuit, int]:
    """
    Create an angle encoding circuit with exactly num_qubits qubits.
    Each qubit gets one U3 gate using up to 3 features as angles.
    
    Args:
    data_point (np.ndarray): Single data point's features
    num_qubits (int): Number of qubits to use (from command line)
    
    Returns:
    Tuple[QuantumCircuit, int]: Quantum circuit and number of qubits used
    """
    # Create circuit with specified number of qubits
    qc = QuantumCircuit(num_qubits)
    
    # Normalize all features
    normalized_features = normalize_features_for_angles(data_point)
    
    # For each qubit, use up to 3 features as U3 angles
    for qubit in range(num_qubits):
        # Get the three features for this qubit's U3 gate
        start_idx = qubit * 3
        if start_idx >= len(normalized_features):
            # If we're out of features, use zeros for angles
            theta, phi, lambda_ = 0.0, 0.0, 0.0
        else:
            # Get up to 3 features, pad with zeros if needed
            theta = normalized_features[start_idx] if start_idx < len(normalized_features) else 0.0
            phi = normalized_features[start_idx + 1] if start_idx + 1 < len(normalized_features) else 0.0
            lambda_ = normalized_features[start_idx + 2] if start_idx + 2 < len(normalized_features) else 0.0
        
        # Apply U3 gate to this qubit
        qc.u(theta, phi, lambda_, qubit)
    
    return qc, num_qubits

def create_angle_encoding_circuits(data: pd.DataFrame, num_qubits: int) -> List[QuantumCircuit]:
    """
    Create angle encoding circuits for all data points.
    
    Args:
    data (pd.DataFrame): Input data
    num_qubits (int): Number of qubits to use (from command line)
    
    Returns:
    List[QuantumCircuit]: List of angle encoding circuits
    """
    circuits = []
    
    print(f"Creating circuits with {num_qubits} qubits")
    print(f"Total features in dataset: {data.shape[1]}")
    print(f"Maximum features that can be encoded: {num_qubits * 3}")
    
    for _, row in data.iterrows():
        circuit, _ = create_angle_encoding_circuit(row.values, num_qubits)
        circuits.append(circuit)
    
    print(f"Total circuits created: {len(circuits)}")
    return circuits