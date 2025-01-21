import argparse
import numpy as np
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from Preprocessing.goldstein_uchida_preprocess import preprocess_goldstein_uchida
from Embedding.angle_encoding import create_angle_encoding_circuit, create_angle_encoding_circuits
from Ansatzes.rx_rz_ansatz import create_encoder_circuit, update_circuit_parameters
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
                             pauli_error, depolarizing_error, thermal_relaxation_error)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Preprocessing and Feature Selection")
    parser.add_argument("num_qubits", type=int, help="Number of qubits to use")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use")
    parser.add_argument("--features_per_qubit", type=int, default=3, help="Number of features to encode per qubit")
    return parser.parse_args()

def create_realistic_noise_model(num_qubits):
    """
    Create a noise model matching IBM's Brisbane quantum computer specifications.
    
    Args:
    num_qubits (int): Number of qubits in the system
    
    Returns:
    NoiseModel: A Qiskit noise model matching Brisbane's error rates
    """
    noise_model = NoiseModel()
    
    # Brisbane specifications
    T1 = 230.42e3  # 230.42 microseconds in nanoseconds
    T2 = 143.41e3  # 143.41 microseconds in nanoseconds
    
    # Gate times (in nanoseconds)
    time_1q = 60    # 1-qubit gate time
    time_2q = 660   # 2-qubit gate time
    time_readout = 1300  # measurement time
    
    # Error rates
    p_sx = 2.274e-4      # single-qubit gate error
    p_cx = 2.903e-3      # two-qubit gate error
    p_readout = 1.38e-2  # readout error
    
    # Add single-qubit gate errors
    for qubit in range(num_qubits):
        # Thermal relaxation error for single-qubit gates
        thermal_error_1q = thermal_relaxation_error(
            T1, T2, time_1q)
        
        # Depolarizing error for single-qubit gates
        depol_error_1q = depolarizing_error(p_sx, 1)
        
        # Combine thermal and depolarizing errors for single-qubit gates
        gate_error_1q = thermal_error_1q.compose(depol_error_1q)
        
        # Add single-qubit gate error
        noise_model.add_quantum_error(gate_error_1q, ["sx"], [qubit])
        
        # Add measurement error
        readout_error = ReadoutError([[1 - p_readout, p_readout], 
                                    [p_readout, 1 - p_readout]])
        noise_model.add_readout_error(readout_error, [qubit])
        
        # Add thermal relaxation during measurement
        meas_thermal_error = thermal_relaxation_error(
            T1, T2, time_readout)
        noise_model.add_quantum_error(meas_thermal_error, ["measure"], [qubit])
    
    # Add two-qubit gate errors
    for q1 in range(num_qubits-1):
        for q2 in range(q1+1, num_qubits):
            # Thermal relaxation error for ECR gate (both qubits)
            thermal_error_q1 = thermal_relaxation_error(
                T1, T2, time_2q)
            thermal_error_q2 = thermal_relaxation_error(
                T1, T2, time_2q)
            thermal_error_2q = thermal_error_q1.expand(thermal_error_q2)
            
            # Depolarizing error for ECR gate
            depol_error_2q = depolarizing_error(p_cx, 2)
            
            # Combine thermal and depolarizing errors for two-qubit gates
            gate_error_2q = thermal_error_2q.compose(depol_error_2q)
            
            # Add two-qubit gate error
            noise_model.add_quantum_error(gate_error_2q, ["cx"], [q1, q2])
    
    return noise_model

#mimics Brisbane noise model
def configure_noisy_simulator(num_qubits):
    """
    Configure the AerSimulator with IBM Brisbane noise settings.
    
    Args:
    num_qubits (int): Number of qubits in the system
    
    Returns:
    AerSimulator: Configured noisy simulator matching Brisbane specifications
    """
    # Create noise model
    noise_model = create_realistic_noise_model(num_qubits)
    
    # Configure simulator
    basis_gates = ['sx', 'rz', 'cx', 'measure']  # Brisbane's basis gates
    simulator = AerSimulator(
        noise_model=noise_model,
        basis_gates=basis_gates,
        coupling_map=[[i, i+1] for i in range(num_qubits-1)]  # Linear nearest-neighbor connectivity
    )
    
    return simulator



def process_iteration(iteration, num_qubits, preprocessed_data, high_risk_indices,
                     simulator, num_bucketruns, angle_encoding_circuits):
    
    print(f"\nStarting iteration {iteration + 1}")
    
    # Create encoder circuit with target of 3 qubits
    target_qubits = 3
    encoder_circuit, encoder_params = create_encoder_circuit(num_qubits, target_qubits)
    # encoder_circuit.draw(output='mpl',filename='encoder_circuit.png')
    num_shots = 4096
    final_results = []
    for _ in range(num_bucketruns):
        # Generate random angles for encoder
        random_angles = np.random.uniform(0, 2*np.pi, len(encoder_params))
        
        # Update the encoder circuit with random angles
        current_circuit = update_circuit_parameters(encoder_circuit, encoder_params, random_angles)
        
        # Run the circuit for each datapoint
        for idx in angle_encoding_circuits.keys():
            full_circuit = angle_encoding_circuits[idx].compose(current_circuit)
            #save circuit drawing from qiskit draw mpl
            # full_circuit.draw(output='mpl',filename='circuit.png')
            result = simulator.run(full_circuit, shots=num_shots).result()
            counts = result.get_counts(full_circuit)
            final_results.append({
                'idx': idx,
                'counts': counts,
                'angles': random_angles
            })

    return {
        'iteration': iteration,
        'measurement_results': final_results,
        'encoder_params': encoder_params,
        'high_risk_indices': high_risk_indices
    }

def main():
    """
    Main function.
    Creates angle encoding circuits once before threading.
    """
    args = parse_arguments()
    num_qubits = args.num_qubits
    num_threads = args.num_threads
    features_per_qubit = args.features_per_qubit
    num_iterations = 1000
    num_bucketruns = 1

    start_time = time.time()

    file_path = './Data/Goldstein_Uchida_datasets/pen-global-unsupervised-ad.csv'

    # Preprocess the data
    preprocessed_data, high_risk_indices, _ = preprocess_goldstein_uchida(file_path)
    
    print(f"Initial dataset size: {len(preprocessed_data)}")
    print(f"Total number of anomalies: {len(high_risk_indices)}")
    print("High risk indices:", high_risk_indices)
    print(f"Number of features: {preprocessed_data.shape[1]}")
    
    # Create angle encoding circuits once for all data points
    angle_encoding_circuits = {}
    for idx, row in preprocessed_data.iterrows():
        circuit, _ = create_angle_encoding_circuit(row.values, num_qubits)  # CORRECT
        angle_encoding_circuits[idx] = circuit
    #draw first encoding circuit
    # angle_encoding_circuits[0].draw(output='mpl',filename='angle_encoding_circuit.png')
    print(f"Created angle encoding circuits for {len(angle_encoding_circuits)} datapoints")

    simulator = AerSimulator()
    
    # Store results for each iteration
    all_results = []
    all_results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for iteration in range(num_iterations):
            future = executor.submit(
                process_iteration,
                iteration,
                num_qubits,
                preprocessed_data,
                high_risk_indices,
                simulator,
                num_bucketruns,
                angle_encoding_circuits
            )
            futures.append(future)

        for future in futures:
            result = future.result()
            with all_results_lock:
                all_results.append(result)

    print("\nAll iterations completed.")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")
    
    with open('results/ensemble_res.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("Results saved to ensemble_res.pkl")
if __name__ == "__main__":
    main()