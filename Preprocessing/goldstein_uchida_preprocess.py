import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

def preprocess_goldstein_uchida(file_path, save_output=False, output_dir='preprocessed_data'):
    """
    Preprocess/Noramlize the Goldstein-Uchida dataset.

    Args:
    file_path (str): Path to the input CSV file.
    save_output (bool): Whether to save the preprocessed data and metadata.
    output_dir (str): Directory to save the preprocessed data and metadata.

    Returns:
    pd.DataFrame: The normalized data.
    list: Indices of anomaly data points.
    dict: Mapping from original indices to preprocessed indices.
    """
    data = pd.read_csv(file_path, header=None)
    
    # remove the label column
    labels = data.iloc[:, -1]
    data = data.iloc[:, :-1]
    
    data = data.astype(float)
    anomaly_indices = labels[labels == 'o'].index.tolist()
    
    # Normalize numerical features
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    normalized_data = range_based_normalize(data)
    
    original_to_preprocessed = dict(zip(data.index, normalized_data.index))
    
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        normalized_data.to_csv(os.path.join(output_dir, f'preprocessed_{base_name}.csv'), index=False)
        np.save(os.path.join(output_dir, f'{base_name}_anomaly_indices.npy'), anomaly_indices)
        with open(os.path.join(output_dir, f'{base_name}_original_to_preprocessed.pkl'), 'wb') as f:
            pickle.dump(original_to_preprocessed, f)
        print(f"Preprocessed data, anomaly indices, and original to preprocessed index mapping saved to {output_dir}")
    
    return normalized_data, anomaly_indices, original_to_preprocessed

def range_based_normalize(data):
    """
    Normalize the data using range-based normalization.

    Args:
    data (pd.DataFrame): The input data to normalize.

    Returns:
    pd.DataFrame: The normalized data.
    """
    num_features = data.shape[1]
    max_value = 1 / num_features
    normalized_data = pd.DataFrame()

    for column in data.columns:
        min_val = data[column].min()
        max_val = data[column].max()
        normalized_data[column] = ((data[column] - min_val) / (max_val - min_val)) * max_value

    return normalized_data