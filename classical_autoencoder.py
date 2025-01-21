import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve, precision_score, recall_score, f1_score, accuracy_score
from Preprocessing.goldstein_uchida_preprocess import preprocess_goldstein_uchida

class AnomalyAutoencoder:
    def __init__(self, input_dim, encoding_dim=8):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = self._build_autoencoder()
        self.threshold = None
        
    def _build_autoencoder(self):
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(32, activation='relu')(input_layer)
        encoded = layers.Dense(16, activation='relu')(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(32, activation='relu')(decoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def fit(self, X_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1):
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return history
    
    def compute_reconstruction_errors(self, X):
        predictions = self.autoencoder.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=1)
        return mse
    
    def set_threshold(self, X_train, contamination=0.1):
        reconstruction_errors = self.compute_reconstruction_errors(X_train)
        self.threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)
        return self.threshold
    
    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        reconstruction_errors = self.compute_reconstruction_errors(X)
        return (reconstruction_errors > self.threshold).astype(int)

def evaluate_model(model, X_test, true_labels, plot=True):
    # Compute reconstruction errors
    reconstruction_errors = model.compute_reconstruction_errors(X_test)
    
    # Get binary predictions using model threshold
    predictions = (reconstruction_errors > model.threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    
    # Calculate ROC curve and AUC for plotting (not used)
    fpr, tpr, _ = roc_curve(true_labels, reconstruction_errors)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve and AUC for plotting
    pr_precision, pr_recall, _ = precision_recall_curve(true_labels, reconstruction_errors)
    pr_auc = auc(pr_recall, pr_precision)
    
    if plot:
        # Plot ROC curve
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Plot Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.plot(pr_recall, pr_precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        #plt.savefig('classical_autoencoder.png')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def plot_reconstruction_errors(model, data_file, preprocessed_data):
    """
    Plot reconstruction errors for each datapoint, sorted from lowest to highest.
    Color bars red for actual anomalies and blue for normal points.
    
    Args:
        model: Trained autoencoder model
        data_file: Path to original CSV file
        preprocessed_data: Preprocessed data used for model
    """
    # Read original data to get labels
    original_data = pd.read_csv(data_file, header=None)
    labels = original_data.iloc[:, -1].values  # Last column contains labels
    
    # Compute reconstruction errors
    reconstruction_errors = model.compute_reconstruction_errors(preprocessed_data)
    
    # Create DataFrame with errors and labels
    error_df = pd.DataFrame({
        'error': reconstruction_errors,
        'is_anomaly': labels == 'o'  # True for anomalies ('o'), False for normal points ('n')
    })
    
    # Sort by error
    error_df = error_df.sort_values('error')
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    
    # Plot bars with different colors based on whether they're anomalies
    normal_mask = ~error_df['is_anomaly']
    anomaly_mask = error_df['is_anomaly']
    
    # Plot normal points (blue)
    plt.bar(np.where(normal_mask)[0], 
            error_df[normal_mask]['error'],
            color='blue',
            alpha=0.6,
            label='Normal')
    
    # Plot anomalies (red)
    plt.bar(np.where(anomaly_mask)[0],
            error_df[anomaly_mask]['error'],
            color='red',
            alpha=0.6,
            label='Anomaly')
    
    plt.axhline(y=model.threshold, color='green', linestyle='--', 
                label=f'Threshold ({model.threshold:.3f})')
    
    plt.xlabel('Data Points (Sorted by Reconstruction Error)')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Errors for All Data Points')
    plt.legend()
    
    # Add summary statistics as text
    total_points = len(error_df)
    total_anomalies = sum(anomaly_mask)
    anomalies_above_threshold = sum((error_df['error'] > model.threshold) & anomaly_mask)
    normal_below_threshold = sum((error_df['error'] <= model.threshold) & normal_mask)
    
    stats_text = f'Total Points: {total_points}\n'
    stats_text += f'True Anomalies: {total_anomalies}\n'
    stats_text += f'Correctly Identified Anomalies: {anomalies_above_threshold}\n'
    stats_text += f'Correctly Identified Normal: {normal_below_threshold}'
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    #plt.savefig('reconstruction_errors.png')

def main():
    # Load and preprocess data
    data_file = 'Data/Goldstein_Uchida_datasets/pen-global-unsupervised-ad.csv'
    data, anomaly_indices, _ = preprocess_goldstein_uchida(data_file)
    
    # Create binary labels (1 for anomaly, 0 for normal)
    labels = np.zeros(len(data))
    labels[anomaly_indices] = 1
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data.values, labels, test_size=0.2, random_state=42,
        stratify=labels
    )
    
    # Initialize and train the model
    model = AnomalyAutoencoder(input_dim=data.shape[1])
    history = model.fit(X_train, epochs=100, batch_size=32)
    
    # Set threshold using training data
    model.set_threshold(X_train, contamination=len(anomaly_indices)/len(data))
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, plot=True)
    
    print("\nModel Performance:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"PR AUC: {metrics['pr_auc']:.3f}")
    
    # Plot reconstruction errors
    plot_reconstruction_errors(model, data_file, data.values)
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
