import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import ParameterVector

def create_ansatz(num_qubits, target_qubits=3, param_prefix='θ'):
    """
    Create a parameterized quantum circuit for the autoencoder ansatz.
    
    Args:
    num_qubits (int): Total number of qubits in the circuit (ceil(num_features/3))
    target_qubits (int): Number of qubits to keep unreset (default 3)
    param_prefix (str): Prefix for parameter names
    """
    qc = QuantumCircuit(num_qubits)
    
    # Calculate number of qubits to reset
    qubits_to_reset = num_qubits - target_qubits
    
    # Calculate parameters needed for rotation gates
    num_parameters = 2 * num_qubits  # Initial RX-RZ layer
    # Add parameters for each compression layer (before reset)
    for i in range(qubits_to_reset):
        num_parameters += 2 * (num_qubits - i)  # RX-RZ pairs for remaining active qubits
        
    params = ParameterVector(param_prefix, num_parameters)
    param_index = 0
    
    # Initial RX-RZ layer on all qubits
    for qubit in range(num_qubits):
        qc.rx(params[param_index], qubit)
        param_index += 1
        qc.rz(params[param_index], qubit)
        param_index += 1
    
    active_qubits = num_qubits
    # Add compression layers
    for layer in range(qubits_to_reset):
        # Add CNOT gates between active qubits
        for q in range(active_qubits - 1):
            qc.cx(q, q + 1)
            
        # Add RX-RZ gates on active qubits
        for qubit in range(active_qubits):
            qc.rx(params[param_index], qubit)
            param_index += 1
            qc.rz(params[param_index], qubit)
            param_index += 1
            
        # Reset the highest-indexed active qubit
        qc.reset(active_qubits - 1)
        active_qubits -= 1

    return qc, params



def create_reset_circuit(num_qubits, compression_level):
    """
    Create a circuit that resets qubits compression_level to num_qubits-1.

    Args:
    num_qubits (int): Total number of qubits in the circuit
    compression_level (int): Number of qubits to compress to.

    Returns:
    QuantumCircuit: Circuit with reset operations.
    """
    reset_circuit = QuantumCircuit(num_qubits)
    for qubit in range(compression_level, num_qubits):
        reset_circuit.reset(qubit)
    return reset_circuit

def create_encoder_circuit(num_qubits, target_qubits=3):
    """
    Create complete encoder circuit with measurements only for unreset qubits.
    
    Args:
    num_qubits (int): Total number of qubits in the circuit
    target_qubits (int): Number of qubits to keep unreset (default 3)
    
    Returns:
    QuantumCircuit: The encoder circuit with reset operations and measurements.
    ParameterVector: The parameters for the encoder RX and RZ gates.
    """
    if target_qubits < 1 or target_qubits > num_qubits:
        raise ValueError(f"Target qubits must be between 1 and {num_qubits}")
        
    encoder, encoder_params = create_ansatz(num_qubits, target_qubits)
    
    # Add classical registers only for unreset qubits
    cr = ClassicalRegister(target_qubits, 'c')
    encoder.add_register(cr)
    
    # Add measurements only for unreset qubits (0 to target_qubits-1)
    for i in range(target_qubits):
        encoder.measure(i, i)
        
    return encoder, encoder_params

def update_circuit_parameters(circuit, encoder_params, new_angles):
    """
    Update the circuit with new angles for encoder.

    Args:
    circuit (QuantumCircuit): The parameterized encoder circuit.
    encoder_params (ParameterVector): The encoder parameters.
    new_angles (list): New angles for the encoder.

    Returns:
    QuantumCircuit: The updated circuit with new angles.
    """
    param_dict = dict(zip(encoder_params, new_angles))
    bound_circuit = circuit.assign_parameters(param_dict)
    return bound_circuit