import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


NUM_HSH_TABLES = 100
NUM_HASHES = 6

class LSH:
    def __init__(self, num_datapoints: int, num_tables: int = NUM_HSH_TABLES, num_hashes: int = NUM_HASHES):
        """
        Initialize LSH with specified number of tables and number of hashes (hyperplanes) per table.
        
        Args:
            num_datapoints (int): Number of datapoints for bucket scaling
            num_tables (int): Number of hash tables (L)
            num_hashes (int): Number of hyperplanes (hashes) per table (K)
        """
        self.num_buckets = max(1, int(num_datapoints * 0.5))
        self.num_tables = num_tables
        self.num_hashes = num_hashes
        self.tables = [defaultdict(list) for _ in range(num_tables)]
        self.hyperplanes = None
    
    def _initialize_hyperplanes(self, vector_dim: int):
        """
        Initialize L tables of K hyperplanes each.
        hyperplanes will be a numpy array of shape (L, K, vector_dim)
        """
        if self.hyperplanes is None:
            # Create L tables each with K hyperplanes
            self.hyperplanes = np.random.randn(self.num_tables, self.num_hashes, vector_dim)
            # Normalize each hyperplane
            norms = np.linalg.norm(self.hyperplanes, axis=2, keepdims=True)
            self.hyperplanes = self.hyperplanes / norms
    
    def _hash_vector_to_table(self, vector: np.ndarray, table_idx: int) -> int:
        """
        Hash a vector using the K hyperplanes for a given table.
        We get a K-bit code: sign(projection) for each hyperplane.
        """
        hyperplanes = self.hyperplanes[table_idx]  # Shape: (K, vector_dim)
        projections = np.dot(hyperplanes, vector)  # Shape: (K,)
        
        # Convert sign pattern to bits
        bits = (projections >= 0).astype(int)  # Convert True/False to 1/0
        # Convert bit array to integer (this will be the bucket key)
        hash_val = 0
        for b in bits:
            hash_val = (hash_val << 1) | b
        return hash_val
    
    def add_vector(self, vector: np.ndarray, index: int):
        """Add a vector to all hash tables."""
        self._initialize_hyperplanes(len(vector))
        
        # Normalize input vector
        vector = vector / np.linalg.norm(vector)
        
        # Hash to each table
        for table_idx in range(self.num_tables):
            bucket = self._hash_vector_to_table(vector, table_idx)
            self.tables[table_idx][bucket].append(index)
    
    def get_bucket_sizes(self) -> Dict[int, List[int]]:
        """Get bucket sizes for each point across all tables."""
        sizes = defaultdict(list)
        for table in self.tables:
            for bucket in table.values():
                bucket_size = len(bucket)
                for idx in bucket:
                    sizes[idx].append(bucket_size)
        return sizes

def counts_to_vector(counts: Dict[str, int]) -> np.ndarray:
    """Convert quantum measurement counts to a vector."""
    first_bitstring = next(iter(counts.keys()))
    num_qubits = len(first_bitstring)
    all_bitstrings = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
    return np.array([counts.get(bitstring, 0) for bitstring in all_bitstrings])

def analyze_iterations(filepath: str = 'results/ensemble_res.pkl', 
                      num_tables: int = NUM_HSH_TABLES, 
                      num_hashes: int = NUM_HASHES):
    """Analyze each iteration using LSH and evaluate anomaly detection performance."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    first_iteration = results[0]['measurement_results']
    unique_indices = set(m['idx'] for m in first_iteration)
    num_datapoints = len(unique_indices)
    high_risk_indices = set(results[0]['high_risk_indices'])
    
    # Calculate anomaly percentage
    anomaly_percentage = len(high_risk_indices) / num_datapoints
    
    print(f"Dataset size: {num_datapoints}")
    print(f"Number of iterations: {len(results)}")
    print(f"Number of anomalies: {len(high_risk_indices)}")
    print(f"Anomaly percentage: {anomaly_percentage:.2%}")
    print(f"LSH parameters: L={num_tables} tables, K={num_hashes} hyperplanes per table")
    
    # Track bucket sizes across iterations
    all_bucket_sizes = defaultdict(list)
    
    for iteration_idx, iteration in enumerate(results):
        print(f"\rProcessing iteration {iteration_idx + 1}/{len(results)}", end="")
        
        # New LSH instance for each iteration
        lsh = LSH(num_datapoints, num_tables, num_hashes)
        
        # Hash all vectors
        for measurement in iteration['measurement_results']:
            idx = measurement['idx']
            vector = counts_to_vector(measurement['counts'])
            lsh.add_vector(vector, idx)
        
        # Record bucket sizes
        sizes = lsh.get_bucket_sizes()
        for idx, size_list in sizes.items():
            all_bucket_sizes[idx].extend(size_list)
    
    print("\nProcessing complete!")
    
    # Calculate median bucket size for each point
    medians = np.zeros(num_datapoints)
    for idx in range(num_datapoints):
        if idx in all_bucket_sizes:
            medians[idx] = np.median(all_bucket_sizes[idx])
    
    # Create true labels array (1 for anomaly, 0 for normal)
    true_labels = np.zeros(num_datapoints)
    true_labels[list(high_risk_indices)] = 1
    
    # Predict anomalies based on threshold
    threshold_idx = int(anomaly_percentage * num_datapoints)
    sorted_indices = np.argsort(medians)  # Sort from lowest (most anomalous) to highest
    predicted_labels = np.zeros(num_datapoints)
    predicted_labels[sorted_indices[:threshold_idx]] = 1  # Label bottom x% as anomalies
    
    # Calculate metrics
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print("\nPerformance Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Original statistics for reference
    anomaly_medians = [medians[i] for i in high_risk_indices]
    normal_medians = [medians[i] for i in range(len(medians)) if i not in high_risk_indices]
    
    print("\nBucket Size Statistics:")
    print(f"Median size for anomalies: {np.median(anomaly_medians):.2f}")
    print(f"Median size for normal points: {np.median(normal_medians):.2f}")
    
    return medians, list(high_risk_indices)
    
def visualize_similarity_scores(scores: np.ndarray, high_risk_indices: List[int], filename: str = 'similarity_scores.png'):
    """Create bar plot of similarity scores with highlighted known anomalies."""
    plt.figure(figsize=(15, 6))
    
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    
    bars = plt.bar(range(len(sorted_scores)), sorted_scores, alpha=0.5, width=1.0)
    
    for i, idx in enumerate(sorted_indices):
        if idx in high_risk_indices:
            bars[i].set_color('red')
            bars[i].set_alpha(0.7)
    
    plt.xlabel('Data Point Index (Sorted by Score)')
    plt.ylabel('Bucket Size (Lower = More Anomalous)')
    plt.title('LSH Bucket Sizes by Data Point')
    
    plt.legend(handles=[
        plt.Rectangle((0,0),1,1, fc='red', alpha=0.7, label='Known Anomalies'),
        plt.Rectangle((0,0),1,1, fc='blue', alpha=0.5, label='Normal Points')
    ])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    medians, high_risk = analyze_iterations(num_tables=NUM_HSH_TABLES, num_hashes=NUM_HASHES)
    visualize_similarity_scores(medians, high_risk)
