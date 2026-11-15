import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm # for progress bar
import torch.nn.functional as F
import pandas as pd
import pickle # For saving/loading scaler
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import the scheduler
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter
import itertools # For generating hyperparameter combinations
import datetime # For unique run names

# Import scikit-learn metrics for threshold optimization and evaluation
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc # For AUC-ROC and AUC-PR
import seaborn as sns


# --- Data Preprocessing (MinMaxScaler) ---
class MinMaxScaler:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, data):
        """
        Calculates min and max values from the data for each feature.
        data: torch.Tensor of shape (N, C) or (N, T, C)
              If (N, T, C), it will compute min/max across N*T for each C.
        """
        if data.ndim == 3: # (N, T, C) -> reshape to (N*T, C)
            data = data.reshape(-1, data.shape[2])
        
        self.min_vals = data.min(dim=0, keepdim=True)[0]
        self.max_vals = data.max(dim=0, keepdim=True)[0]

    def transform(self, data):
        """
        Applies Min-Max scaling using the fitted min and max values.
        Scales data to the [0, 1] range.
        data: torch.Tensor of shape (N, C) or (N, T, C)
        """
        if self.min_vals is None or self.max_vals is None:
            raise RuntimeError("MinMaxScaler not fitted. Call .fit() first.")
        
        # Add a small epsilon to the denominator to prevent division by zero
        denominator = (self.max_vals - self.min_vals)
        denominator[denominator == 0] = 1.0 # Handle constant features by setting denominator to 1

        if data.ndim == 3:
            return (data - self.min_vals.unsqueeze(1)) / denominator.unsqueeze(1)
        else: # (N, C)
            return (data - self.min_vals) / denominator

    def inverse_transform(self, data):
        """
        Reverses the Min-Max scaling.
        data: torch.Tensor of shape (N, C) or (N, T, C)
        """
        if self.min_vals is None or self.max_vals is None:
            raise RuntimeError("MinMaxScaler not fitted. Call .fit() first.")

        denominator = (self.max_vals - self.min_vals)
        denominator[denominator == 0] = 1.0 

        if data.ndim == 3:
            return (data * denominator.unsqueeze(1)) + self.min_vals.unsqueeze(1)
        else:
            return (data * denominator) + self.min_vals

# --- Configuration (Hyperparameters) ---
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42 # Seed for reproducibility of data splits. Crucial for consistent train/test sets!
        self.vae_epochs = 500 
        
        # --- Fixed parameters or default values for hyperparameter search ---
        self.batch_size = 8 
        self.learning_rate = 1e-3
        self.hidden_size_vae = 256 # Hidden size for VAE's LSTM
        self.latent_dim = 128 # Dimension of the VAE's latent space
        self.vae_beta = 1.0 # Beta value for VAE loss: Recon_Loss + beta * KL_Loss

        # self.reconstruction_error_threshold_quantile = 0.95 # This will no longer be directly used for threshold setting

        # --- Early Stopping Parameters ---
        self.patience_vae = 25 
        self.min_delta = 1e-4 

        # --- Learning Rate Scheduler Parameters ---
        self.scheduler_patience_vae = 5 # How many epochs to wait for improvement before reducing LR
        self.scheduler_factor = 0.5    # Factor by which the learning rate will be reduced (e.g., new_lr = old_lr * factor)
        self.min_lr = 1e-6           # Minimum learning rate

        # --- Data Specific Configuration ---
        self.normal_feature_folder = os.environ.get("CAG_NORMAL_FEATURES", "data/normal/features")
        self.normal_annotation_folder = os.environ.get("CAG_NORMAL_ANNOTATIONS", "data/normal/annotations")
        
        self.anomaly_feature_folder = os.environ.get("CAG_ANOMALY_FEATURES", "data/anomaly/features")
        self.anomaly_annotation_folder = os.environ.get("CAG_ANOMALY_ANNOTATIONS", "data/anomaly/annotations")

        self.seq_len = 574 # Fixed sequence length for your data (number of 10-sec segments per procedure)
        self.feature_dim = 2048 # Dimension of each 10-sec segment's feature vector
        self.padding = 'zero' # or 'repeat_last'

        self.data_preprocessor = MinMaxScaler() # Initialize the MinMaxScaler here!

        # --- Hyperparameter Search Log Folder ---
        # This will be used as the base, and a timestamped subfolder will be created for each search session
        self.base_log_dir = os.environ.get("CAG_RUNS_DIR", "runs")


# Initialize config (used as a base template)
base_config = Config()
# Set manual seeds for reproducibility. This is crucial for consistent data splits.
torch.manual_seed(base_config.seed)
np.random.seed(base_config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(base_config.seed)

# --- VAE Model Definition ---
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        # Encoder: processes the (seq_len, input_dim) sequence
        self.encoder_lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Decoder: reconstructs the (seq_len, input_dim) sequence from a single latent vector
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_size, batch_first=True)
        # Apply sigmoid to output layer to constrain reconstructions to [0, 1] range, matching MinMaxScaler output
        self.fc_reconstruct = nn.Linear(hidden_size, input_dim) 

    def encode(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        _, (hidden, _) = self.encoder_lstm(x)
        # hidden shape: (1, batch_size, hidden_size) -> take last hidden state
        hidden = hidden.squeeze(0) # (batch_size, hidden_size)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, sequence_length):
        # z shape: (batch_size, latent_dim)
        # We need to expand z to be a sequence for the decoder LSTM
        # Repeat z for each timestep in the sequence
        z_expanded = z.unsqueeze(1).repeat(1, sequence_length, 1) # (batch_size, sequence_length, latent_dim)
        output, _ = self.decoder_lstm(z_expanded)
        reconstruction = torch.sigmoid(self.fc_reconstruct(output)) # Apply sigmoid for [0, 1] output, matching data scaling
        return reconstruction

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        sequence_length = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, sequence_length)
        return reconstruction, mu, logvar

# --- VAE Loss Function (with Beta) ---
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0): 
    # recon_x and x shape: (batch_size, sequence_length, input_dim)
    # mu and logvar shape: (batch_size, latent_dim)

    # Reconstruction loss (MSE)
    # Sum over features and sequence length, then average over batch
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + beta * kl_div, recon_loss, kl_div 

# --- Data Loading and Preprocessing ---
class SurgicalProcedureDataset(Dataset):
    def __init__(self, feature_folder, annotation_folder, seq_len=None, padding='zero', preprocessor=None,feature_dim=512):
        self.feature_folder = feature_folder
        self.annotation_folder = annotation_folder
        self.files = sorted([
            os.path.join(feature_folder, fname)
            for fname in os.listdir(feature_folder)
            if fname.endswith('.pkl')
        ])
        self.seq_len = seq_len
        self.padding = padding
        self.preprocessor = preprocessor 
        self.feature_dim=feature_dim

    def __len__(self):
        return len(self.files)

    def load_sequence(self, file_path):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # The ID is the base_name without the .xlsx extension (which is already handled by os.path.splitext)
        # So, base_name itself is the desired ID.
        annotation_path = os.path.join(self.annotation_folder, base_name + ".csv")

        index1 = 0
        index2 = None

        # Load annotations and determine index1 and index2
        if os.path.exists(annotation_path):
            df = pd.read_csv(annotation_path)
            if "Phase" in df.columns:
                a_indices = df[df["Phase"] == "A"].index
                if len(a_indices) > 0:
                    index1 = int(a_indices[-1] / 10)
                m_indices = df[df["Phase"] == "M"].index
                if len(m_indices) > 0:
                    index2 = int(m_indices[0] / 10)
                    
        # Load the .pkl feature sequence
        with open(file_path, 'rb') as f:
            sequence = pickle.load(f)

        # Slice sequence between index1 and index2 (if defined)
        if index2 is not None:
            sequence = sequence[index1:index2]
        else:
            sequence = sequence[index1:]

        # Pad or truncate if necessary
        T_new = len(sequence)
        if self.seq_len is not None:
            if T_new > self.seq_len:
                sequence = sequence[:self.seq_len]
            elif T_new < self.seq_len:
                if self.padding == 'zero':
                    pad = np.zeros((self.seq_len - T_new, self.feature_dim), dtype=np.float32)
                elif self.padding == 'repeat_last':
                    if T_new > 0:
                        pad = np.repeat(sequence[-1][None, :], self.seq_len - T_new, axis=0)
                    else:  
                        pad = np.zeros((self.seq_len - T_new, self.feature_dim), dtype=np.float32)
                sequence = np.vstack([sequence, pad])

        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        if self.preprocessor:
            # Ensure preprocessor can handle 2D data (seq_len, feature_dim)
            # It's assumed preprocessor.transform expects a 2D array and returns 2D.
            sequence_tensor = self.preprocessor.transform(sequence_tensor)
            
        return sequence_tensor, base_name # Return both the tensor and the ID

    def __getitem__(self, idx):
        file_path = self.files[idx]
        sequence_tensor, data_id = self.load_sequence(file_path) # Get both
        return sequence_tensor, data_id # Return both

# --- Helper functions for data loaders (MODIFIED FOR NEW SPLITS) ---
def get_normal_data_loaders(config):
    """
    Loads normal data, fits the scaler, and splits into VAE train/val,
    and normal calibration/test sets.
    """
    # Create a temporary dataset just to load all data for scaler fitting
    # Only fit scaler if it's not already loaded (e.g., in reproduction mode)
    if config.data_preprocessor.min_vals is None or config.data_preprocessor.max_vals is None:
        temp_normal_dataset = SurgicalProcedureDataset(
            feature_folder=config.normal_feature_folder,
            annotation_folder=config.normal_annotation_folder,
            seq_len=config.seq_len,
            padding=config.padding,
            preprocessor=None,
            feature_dim=config.feature_dim
        )

        all_normal_data_raw = []
        # Loop over indices and discard the ID for scaler fitting
        for i in tqdm(range(len(temp_normal_dataset)), desc="Loading all normal data to fit scaler"):
            sequence_data, _ = temp_normal_dataset[i] # Get sequence data, ignore ID
            all_normal_data_raw.append(sequence_data)

        all_normal_data_raw_tensor = torch.cat([d.unsqueeze(0) for d in all_normal_data_raw], dim=0)

        print("Fitting MinMaxScaler on normal training data...")
        config.data_preprocessor.fit(all_normal_data_raw_tensor)
        print(f"Scaler fitted. Min shape: {config.data_preprocessor.min_vals.shape}, Max shape: {config.data_preprocessor.max_vals.shape}")
    else:
        print("MinMaxScaler already fitted (or loaded). Skipping fitting step.")


    # Create the full dataset with the fitted preprocessor
    full_normal_dataset_scaled = SurgicalProcedureDataset(
        feature_folder=config.normal_feature_folder,
        annotation_folder=config.normal_annotation_folder,
        seq_len=config.seq_len,
        padding=config.padding,
        preprocessor=config.data_preprocessor,
            feature_dim=config.feature_dim
    )

    # Define split ratios for normal data
    total_size = len(full_normal_dataset_scaled)
    # Recommended split: 50% train_vae, 10% val_vae, 20% calibration_normal, 20% test_normal
    train_vae_size = int(0.50 * total_size)
    val_vae_size = int(0.10 * total_size)
    
    remaining_size = total_size - train_vae_size - val_vae_size
    calibration_normal_size = int(0.5 * remaining_size) # Half of remaining for calibration
    test_normal_size = remaining_size - calibration_normal_size # Other half for test

    # Perform reproducible random split
    train_vae_dataset, val_vae_dataset, calibration_normal_dataset, test_normal_dataset = torch.utils.data.random_split(
        full_normal_dataset_scaled,
        [train_vae_size, val_vae_size, calibration_normal_size, test_normal_size],
        generator=torch.Generator().manual_seed(config.seed) # Ensure reproducibility of splits
    )

    train_vae_loader = DataLoader(train_vae_dataset, batch_size=config.batch_size, shuffle=True)
    val_vae_loader = DataLoader(val_vae_dataset, batch_size=config.batch_size, shuffle=False)

    return train_vae_loader, val_vae_loader, calibration_normal_dataset, test_normal_dataset

def get_anomaly_data_loader(config):
    """
    Loads anomaly data and splits into calibration and test sets.
    """
    full_anomaly_dataset_scaled = SurgicalProcedureDataset(
        feature_folder=config.anomaly_feature_folder,
        annotation_folder=config.anomaly_annotation_folder,
        seq_len=config.seq_len,
        padding=config.padding,
        preprocessor=config.data_preprocessor, # Use the same fitted preprocessor
            feature_dim=config.feature_dim

    )

    # Define split ratios for anomaly data: e.g., 50% calibration, 50% test
    total_size = len(full_anomaly_dataset_scaled)
    calibration_size = int(0.5 * total_size)
    test_size = total_size - calibration_size

    # Perform reproducible random split
    calibration_anomaly_dataset, test_anomaly_dataset = torch.utils.data.random_split(
        full_anomaly_dataset_scaled,
        [calibration_size, test_size],
        generator=torch.Generator().manual_seed(config.seed) # Ensure reproducibility of splits
    )

    return calibration_anomaly_dataset, test_anomaly_dataset

# --- Optimal Threshold Finding Function ---
def find_optimal_threshold_f1(normal_scores, anomaly_scores):
    """
    Finds the optimal threshold by maximizing the F1-score on a calibration set.

    Args:
        normal_scores (np.array): Anomaly scores for normal samples in the calibration set.
        anomaly_scores (np.array): Anomaly scores for anomaly samples in the calibration set.

    Returns:
        float: The optimal threshold.
        dict: Dictionary containing best metrics (f1, precision, recall) at that threshold.
    """
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    true_labels = np.concatenate([np.zeros_like(normal_scores), np.ones_like(anomaly_scores)]) # 0 for normal, 1 for anomaly

    # Filter out NaN values from scores before processing
    valid_scores_mask = ~np.isnan(all_scores)
    all_scores_filtered = all_scores[valid_scores_mask]
    true_labels_filtered = true_labels[valid_scores_mask]


    if len(np.unique(true_labels_filtered)) < 2 or len(all_scores_filtered) == 0:
        print("Warning: Calibration set contains only one class or no valid data (all NaNs). Cannot optimize F1-score.")
        # Fallback to a percentile if only normal data is present in calibration
        if len(normal_scores) > 0 and len(anomaly_scores) == 0:
            print("Using 95th percentile of normal scores as fallback threshold.")
            # Ensure normal_scores used here are also finite
            finite_normal_scores = normal_scores[~np.isnan(normal_scores)]
            if len(finite_normal_scores) > 0:
                fallback_threshold = np.quantile(finite_normal_scores, 0.95)
            else:
                fallback_threshold = 0.0 # Default if even normal scores are all NaN or empty
            return fallback_threshold, {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        else: # Handle cases with no data or only anomalies (shouldn't happen if calibrated right)
            return 0.0, {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    # Sort scores to iterate through unique values as potential thresholds
    thresholds = np.sort(np.unique(all_scores_filtered))

    best_f1 = -1
    optimal_threshold = None
    best_precision = 0
    best_recall = 0

    for t in thresholds:
        predicted_labels = (all_scores_filtered > t).astype(int)
        
        # F1-score for binary classification where 1 is the positive class (anomaly)
        f1 = f1_score(true_labels_filtered, predicted_labels, pos_label=1, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = t
            best_precision = precision_score(true_labels_filtered, predicted_labels, pos_label=1, zero_division=0)
            best_recall = recall_score(true_labels_filtered, predicted_labels, pos_label=1, zero_division=0)

    if optimal_threshold is None: # Fallback if no valid threshold found (e.g., all scores are identical)
        optimal_threshold = np.mean(all_scores_filtered) # Or some other sensible default
        print("Warning: Optimal threshold could not be found via F1-score maximization. Falling back to mean score.")

    print(f"Optimal Threshold (F1-score maximization): {optimal_threshold:.4f}")
    print(f"  Best F1-Score: {best_f1:.4f}")
    print(f"  Best Precision: {best_precision:.4f}")
    print(f"  Best Recall: {best_recall:.4f}")

    return optimal_threshold, {'f1': best_f1, 'precision': best_precision, 'recall': best_recall}


# --- Evaluation Metrics Function ---
def evaluate_performance(normal_test_scores, anomaly_test_scores, threshold):
    """
    Evaluates the anomaly detection performance on test sets using a fixed threshold.

    Args:
        normal_test_scores (np.array): Anomaly scores for test normal samples.
        anomaly_test_scores (np.array): Anomaly scores for test anomaly samples.
        threshold (float): The fixed anomaly threshold.

    Returns:
        dict: A dictionary of performance metrics.
    """
    # Combine scores and labels
    all_scores = np.concatenate([normal_test_scores, anomaly_test_scores])
    true_labels = np.concatenate([np.zeros_like(normal_test_scores), np.ones_like(anomaly_test_scores)])
    
    # Filter out NaN scores and their corresponding labels
    valid_mask = ~np.isnan(all_scores)
    all_scores_filtered = all_scores[valid_mask]
    true_labels_filtered = true_labels[valid_mask]
    
    # Predicted labels based on the threshold
    predicted_labels = (all_scores_filtered > threshold).astype(int)

    # Initialize metrics with 0.0 or 0 for cases where calculation might not be possible
    accuracy, precision, recall, f1, auc_roc, auc_pr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
    fpr, tpr = 0.0, 0.0
    cm = np.zeros((2,2), dtype=int) # Default empty confusion matrix

    # Only attempt metric calculation if there is valid data
    if len(true_labels_filtered) > 0:
        accuracy = accuracy_score(true_labels_filtered, predicted_labels)
        precision = precision_score(true_labels_filtered, predicted_labels, pos_label=1, zero_division=0)
        recall = recall_score(true_labels_filtered, predicted_labels, pos_label=1, zero_division=0)
        f1 = f1_score(true_labels_filtered, predicted_labels, pos_label=1, zero_division=0)
        cm = confusion_matrix(true_labels_filtered, predicted_labels)

        # Specific counts
        # These are computed from the filtered scores and labels to avoid NaNs
        true_positives = np.sum((all_scores_filtered > threshold) & (true_labels_filtered == 1))
        false_positives = np.sum((all_scores_filtered > threshold) & (true_labels_filtered == 0))
        true_negatives = np.sum((all_scores_filtered <= threshold) & (true_labels_filtered == 0))
        false_negatives = np.sum((all_scores_filtered <= threshold) & (true_labels_filtered == 1))

        # Rates
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0 # Recall

        # Calculate AUC-ROC
        if len(np.unique(true_labels_filtered)) > 1 and len(np.unique(all_scores_filtered)) > 1:
            auc_roc = roc_auc_score(true_labels_filtered, all_scores_filtered)
        else:
            print("Warning: Cannot compute AUC-ROC, either only one class or all scores are the same (after filtering NaNs).")

        # Calculate AUC-PR
        # precision_recall_curve can handle filtered scores directly
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels_filtered, all_scores_filtered)
        # Check if precision_curve and recall_curve are valid before calculating AUC
        if len(precision_curve) > 1 and len(recall_curve) > 1:
            auc_pr = auc(recall_curve, precision_curve)
        else:
            print("Warning: Cannot compute AUC-PR, precision/recall curve is degenerate (after filtering NaNs).")


    print(f"\n--- Evaluation Results (Threshold: {threshold:.4f}) ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision (Anomaly): {precision:.4f}")
    print(f"Overall Recall (Anomaly): {recall:.4f}")
    print(f"Overall F1-Score (Anomaly): {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}") # Added AUC-ROC
    print(f"AUC-PR: {auc_pr:.4f}")   # Added AUC-PR
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"  True Positives (Anomalies Correctly Detected): {true_positives}")
    print(f"  False Positives (Normal Misclassified as Anomaly): {false_positives}")
    print(f"  True Negatives (Normal Correctly Classified): {true_negatives}")
    print(f"  False Negatives (Anomalies Missed): {false_negatives}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"True Positive Rate (TPR/Recall): {tpr:.4f}")

    return {
        'overall_accuracy': accuracy,
        'overall_precision': precision,
        'overall_recall': recall,
        'overall_f1': f1,
        'cm': cm.tolist(), # Convert to list for easier logging/saving
        'tp': true_positives,
        'fp': false_positives,
        'tn': true_negatives,
        'fn': false_negatives,
        'fpr': fpr,
        'tpr': tpr,
        'auc_roc': auc_roc, # Added
        'auc_pr': auc_pr    # Added
    }

# --- Plotting function for loss curves (optional, can use TensorBoard instead) ---
def plot_loss_curves(train_losses, val_losses, title):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Training Function (Only VAE with LR Scheduler and TensorBoard Logging) ---
def train_vae(vae_model, train_loader, val_loader, config, writer, run_name):
    optimizer = optim.Adam(vae_model.parameters(), lr=config.learning_rate)
    # Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',          # Monitor validation loss for reduction (minimize loss)
        factor=config.scheduler_factor, # Factor by which the learning rate will be reduced
        patience=config.scheduler_patience_vae, # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=config.min_lr, # A lower bound on the learning rate
    )

    vae_model.train()
    vae_model.to(config.device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses = [] 
    val_losses = []   

    print(f"\n--- Training VAE for {config.vae_epochs} epochs (Run: {run_name}) ---")
    for epoch in range(config.vae_epochs):
        # --- Training Loop ---
        vae_model.train() 
        total_train_loss = 0
        total_train_recon_loss = 0
        total_train_kl_div = 0
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.vae_epochs} (VAE Train)")): # Unpack data, ignore ID
            data = data.to(config.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model(data)
            
            # Pass the beta value to the loss function
            loss, recon_loss, kl_div = vae_loss_function(recon_batch, data, mu, logvar, beta=config.vae_beta)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_recon_loss += recon_loss.item()
            total_train_kl_div += kl_div.item()
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss) 
        avg_train_recon_loss = total_train_recon_loss / len(train_loader.dataset)
        avg_train_kl_div = total_train_kl_div / len(train_loader.dataset)

        # --- Validation Loop ---
        vae_model.eval() 
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.vae_epochs} (VAE Val)")): # Unpack data, ignore ID
                data = data.to(config.device)
                recon_batch, mu, logvar = vae_model(data)
                # Pass the beta value to the loss function for validation
                loss, _, _ = vae_loss_function(recon_batch, data, mu, logvar, beta=config.vae_beta)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss) 
        
        print(f"Epoch {epoch+1} VAE Train Loss: {avg_train_loss:.4f}, Recon Loss: {avg_train_recon_loss:.4f}, KL Div: {avg_train_kl_div:.4f} | VAE Val Loss: {avg_val_loss:.4f}")

        # --- TensorBoard Logging (Epoch-wise) ---
        writer.add_scalar('Loss/VAE_Train_Total', avg_train_loss, epoch)
        writer.add_scalar('Loss/VAE_Train_Recon', avg_train_recon_loss, epoch)
        writer.add_scalar('Loss/VAE_Train_KL', avg_train_kl_div, epoch)
        writer.add_scalar('Loss/VAE_Val_Total', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)


        # Step the learning rate scheduler
        scheduler.step(avg_val_loss)

        # --- Early Stopping Logic ---
        # The best model will be saved with a unique name for this run
        current_vae_model_path = os.path.join(writer.log_dir, "vae_model_best_run.pth") # Use writer.log_dir directly
        current_scaler_path = os.path.join(writer.log_dir, "scaler_best_run.pkl") # Path to save scaler

        if avg_val_loss < best_val_loss - config.min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Ensure the directory exists before saving the model and scaler
            os.makedirs(os.path.dirname(current_vae_model_path), exist_ok=True)
            torch.save(vae_model.state_dict(), current_vae_model_path) 
            # Save the fitted scaler
            with open(current_scaler_path, 'wb') as f:
                pickle.dump(config.data_preprocessor, f)
            print(f"VAE model and scaler improved. Saving to {current_vae_model_path} and {current_scaler_path}")
        else:
            epochs_no_improve += 1
            print(f"VAE model did not improve. Epochs without improvement: {epochs_no_improve}/{config.patience_vae}")
            if epochs_no_improve >= config.patience_vae:
                print(f"Early stopping triggered for VAE after {epoch+1} epochs.")
                break 
    
    # Load the best model after training (even if early stopping occurred)
    if os.path.exists(current_vae_model_path):
        print(f"Loading best VAE model from {current_vae_model_path} for evaluation.")
        vae_model.load_state_dict(torch.load(current_vae_model_path, map_location=config.device))
        # Load the best scaler
        if os.path.exists(current_scaler_path):
            with open(current_scaler_path, 'rb') as f:
                config.data_preprocessor = pickle.load(f)
            print(f"Loaded best scaler from {current_scaler_path}.")
        else:
            print(f"Warning: Best scaler not found at {current_scaler_path}. Using the last fitted scaler.")
    else:
        print("No best VAE model saved (perhaps val loss never improved significantly). Using last trained state.")
        # If no best model was saved (e.g., very short training, no improvement), ensure it's still on device
        vae_model.to(config.device)

    return best_val_loss # Return the best validation loss for hparams logging


# --- Anomaly Score Calculation Function ---
def calculate_anomaly_scores_vae_only_old(vae_model, dataset_to_evaluate, config):
    vae_model.eval()
    vae_model.to(config.device)

    all_vae_recon_errors = []
    all_ids = [] # List to store the data IDs
    
    # Use DataLoader for efficient batch processing during anomaly score calculation
    # Ensure shuffle=False for evaluation to maintain order for ID mapping
    eval_loader = DataLoader(dataset_to_evaluate, batch_size=config.batch_size, shuffle=False) 

    with torch.no_grad():
        for batch_idx, (sequences, ids) in enumerate(tqdm(eval_loader, desc="Calculating VAE Anomaly Scores")):
            sequences = sequences.to(config.device)

            # --- VAE Reconstruction Error ---
            recon_sequences, _, _ = vae_model(sequences)
            
            # Sum of MSE over (seq_len, 512) for each sample in the batch
            # and then convert to numpy
            batch_recon_errors = F.mse_loss(recon_sequences, sequences, reduction='none').sum(dim=[1, 2]).cpu().numpy()
            
            # Check for NaNs and handle them
            # Replace NaNs with a large finite number (e.g., max float value)
            # This treats any uncomputable score as a definitive anomaly, preventing propagation.
            if np.isnan(batch_recon_errors).any():
                print(f"Warning: NaN detected in VAE reconstruction error for batch {batch_idx}. Replacing with {np.finfo(np.float32).max}.")
                batch_recon_errors[np.isnan(batch_recon_errors)] = np.finfo(np.float32).max 
            
            all_vae_recon_errors.extend(batch_recon_errors)
            all_ids.extend(ids) # IDs are already a list/tuple from dataset

    return np.array(all_vae_recon_errors), all_ids # Return scores and IDs
# --- Anomaly Score Calculation Function ---
def calculate_anomaly_scores_vae_only(vae_model, dataset_to_evaluate, config):
    vae_model.eval()
    vae_model.to(config.device)

    all_vae_overall_recon_errors = [] # Overall anomaly score per procedure (sum over all dimensions)
    all_ids = [] # List to store the data IDs
    all_avg_clip_errors = [] # List to store (1, M) average error per clip for each procedure
    
    # Use DataLoader for efficient batch processing during anomaly score calculation
    # Ensure shuffle=False for evaluation to maintain order for ID mapping
    eval_loader = DataLoader(dataset_to_evaluate, batch_size=config.batch_size, shuffle=False) 

    with torch.no_grad():
        for batch_idx, (sequences, ids) in enumerate(tqdm(eval_loader, desc="Calculating VAE Anomaly Scores")):
            sequences = sequences.to(config.device)

            # --- VAE Reconstruction Error ---
            recon_sequences, _, _ = vae_model(sequences)
            
            # 1. Overall reconstruction error per procedure (sum over seq_len and feature_dim)
            batch_overall_recon_errors = F.mse_loss(recon_sequences, sequences, reduction='none').sum(dim=[1, 2]).cpu().numpy()
            
            # 2. Average reconstruction error per clip (mean over feature_dim)
            # Input to this step: F.mse_loss(recon_sequences, sequences, reduction='none') has shape (batch_size, seq_len, feature_dim)
            # After .mean(dim=2), the shape for each item in the batch becomes (seq_len,)
            # So, batch_avg_clip_errors will have a shape of (batch_size, seq_len)
            batch_avg_clip_errors = F.mse_loss(recon_sequences, sequences, reduction='none').sum(dim=2).cpu().numpy()

            # Check for NaNs and handle them for both types of errors
            # Replace NaNs with a large finite number (e.g., max float value)
            # This treats any uncomputable score as a definitive anomaly, preventing propagation.
            if np.isnan(batch_overall_recon_errors).any():
                print(f"Warning: NaN detected in overall VAE reconstruction error for batch {batch_idx}. Replacing with {np.finfo(np.float32).max}.")
                batch_overall_recon_errors[np.isnan(batch_overall_recon_errors)] = np.finfo(np.float32).max 
            
            if np.isnan(batch_avg_clip_errors).any():
                print(f"Warning: NaN detected in average clip VAE reconstruction error for batch {batch_idx}. Replacing with {np.finfo(np.float32).max}.")
                batch_avg_clip_errors[np.isnan(batch_avg_clip_errors)] = np.finfo(np.float32).max 
            
            all_vae_overall_recon_errors.extend(batch_overall_recon_errors)
            all_ids.extend(ids) # IDs are already a list/tuple from dataset
            # Extend with each (1, M) array from the batch
            for i in range(batch_avg_clip_errors.shape[0]):
                all_avg_clip_errors.append(batch_avg_clip_errors[i:i+1, :]) # Keep as (1, M) shape, where M=seq_len

    return np.array(all_vae_overall_recon_errors), all_ids, all_avg_clip_errors # Return scores, IDs, and per-clip errors


# --- Plotting function for anomaly scores (MODIFIED) ---
def plot_anomaly_scores_split(normal_scores, anomaly_scores, normal_ids, anomaly_ids, threshold, title="Anomaly Scores"):
    """
    Plots anomaly scores for normal and anomaly data separately, with a common threshold,
    and displays the IDs of the data points.
    Normal data is shown first, then anomaly data.

    Args:
        normal_scores (np.array): Anomaly scores for normal samples.
        anomaly_scores (np.array): Anomaly scores for anomaly samples.
        normal_ids (list): List of IDs corresponding to normal_scores.
        anomaly_ids (list): List of IDs corresponding to anomaly_scores.
        threshold (float): The detection threshold.
        title (str): Plot title.
    """
    plt.figure(figsize=(18, 7)) # Wider figure to accommodate two groups

    # Calculate y-offset for text labels
    # Handle cases where scores might be empty
    all_scores_concat = []
    if normal_scores.size > 0:
        all_scores_concat.append(normal_scores)
    if anomaly_scores.size > 0:
        all_scores_concat.append(anomaly_scores)

    if all_scores_concat:
        min_overall_score = np.min(np.concatenate(all_scores_concat))
        max_overall_score = np.max(np.concatenate(all_scores_concat))
    else:
        min_overall_score = 0
        max_overall_score = 1 # Default range if no scores

    y_range = max_overall_score - min_overall_score
    y_text_offset = 0.05 * y_range if y_range > 0 else 0.01 # Ensure non-zero offset

    # Plot normal scores
    plt.plot(np.arange(len(normal_scores)), normal_scores, 'b.', alpha=0.6, label='Normal Scores')
    normal_anomalies_fp = normal_scores > threshold # False Positives
    plt.scatter(np.where(normal_anomalies_fp)[0], normal_scores[normal_anomalies_fp],
                color='red', marker='x', s=100, linewidth=2, label='Normal (False Positive)')

    # Add IDs for normal points
    for i, score in enumerate(normal_scores):
        plt.text(i, score + y_text_offset, normal_ids[i],
                 fontsize=8, ha='center', va='bottom', color='blue', rotation=45)

    # Plot anomaly scores
    # Adjust x-axis for anomaly scores to appear after normal scores
    anomaly_start_idx = len(normal_scores)
    plt.plot(np.arange(anomaly_start_idx, anomaly_start_idx + len(anomaly_scores)), anomaly_scores, 'g.', alpha=0.6, label='Anomaly Scores')
    anomaly_detected_tp = anomaly_scores > threshold # True Positives
    plt.scatter(np.where(anomaly_detected_tp)[0] + anomaly_start_idx, anomaly_scores[anomaly_detected_tp],
                color='darkgreen', marker='o', s=100, linewidth=2, label='Anomaly (True Positive)')
    
    anomaly_missed_fn = anomaly_scores <= threshold # False Negatives
    plt.scatter(np.where(anomaly_missed_fn)[0] + anomaly_start_idx, anomaly_scores[anomaly_missed_fn],
                color='orange', marker='s', s=100, linewidth=2, label='Anomaly (False Negative)')

    # Add IDs for anomaly points
    for i, score in enumerate(anomaly_scores):
        plt.text(i + anomaly_start_idx, score + y_text_offset, anomaly_ids[i],
                 fontsize=8, ha='center', va='bottom', color='green', rotation=45)


    plt.axhline(y=threshold, color='purple', linestyle='--', label=f'Optimal Threshold ({threshold:.2f})')
    plt.title(title)
    plt.xlabel("Sample Index (Normal then Anomaly)")
    plt.ylabel("Anomaly Score (Reconstruction Error)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_combined_anomaly_scores(all_run_data, save_path=None):
    """
    Plots anomaly scores for multiple runs on a single figure,
    with different colors/markers for each run and their respective thresholds.

    Args:
        all_run_data (dict): A dictionary where keys are run names (e.g., 'Pose', 'Video', 'Late Fusion')
                             and values are dictionaries containing:
                             'test_normal_scores', 'test_anomaly_scores', 'optimal_threshold'.
        save_path (str, optional): Path to save the plot. Defaults to None (show plot).
    """
    plt.figure(figsize=(15, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    line_styles = ['-', '--', ':', '-.']

    for i, (run_name, data) in enumerate(all_run_data.items()):
        normal_scores = data['test_normal_scores']
        anomaly_scores = data['test_anomaly_scores']
        threshold = data['optimal_threshold']

        # Plot normal scores
        plt.scatter(
            np.arange(len(normal_scores)),
            normal_scores,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=20,
            alpha=0.6,
            label=f'{run_name} Normal Scores' if i == 0 else "" # Label only once per run type
        )
        # Plot anomaly scores
        plt.scatter(
            np.arange(len(normal_scores), len(normal_scores) + len(anomaly_scores)),
            anomaly_scores,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            facecolors='none', # Hollow markers for anomalies
            edgecolors=colors[i % len(colors)],
            s=40,
            linewidth=1.5,
            label=f'{run_name} Anomaly Scores' if i == 0 else ""
        )
        # Plot threshold line
        plt.axhline(
            y=threshold,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2,
            label=f'{run_name} Threshold ({threshold:.2f})'
        )

    plt.title('Comparative Anomaly Scores on Test Data', fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    if save_path:
        plt.savefig(save_path)
        print(f"Combined anomaly scores plot saved to {save_path}")
    plt.show()
    plt.close()

def plot_confusion_matrix(true_labels, predictions, run_name, save_path=None):
    """
    Plots a confusion matrix.

    Args:
        true_labels (np.array): True labels (0 for normal, 1 for anomaly).
        predictions (np.array): Predicted labels (0 for normal, 1 for anomaly).
        run_name (str): Name of the run for the plot title.
        save_path (str, optional): Path to save the plot. Defaults to None (show plot).
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['True Normal', 'True Anomaly'])
    plt.title(f'Confusion Matrix - {run_name}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix for {run_name} saved to {save_path}")
    plt.show()
    plt.close()

def plot_combined_metrics(all_run_metrics, save_path=None):
    """
    Presents key metrics for all runs in a comparative bar chart and a table.

    Args:
        all_run_metrics (dict): A dictionary where keys are run names and values are
                                dictionaries of metrics (e.g., 'f1', 'precision', 'recall', 'accuracy').
        save_path (str, optional): Path to save the plot. Defaults to None (show plot).
    """
    metric_names = ['f1', 'precision', 'recall', 'overall_accuracy', 'auc_roc', 'auc_pr']
    run_names = list(all_run_metrics.keys())

    # Prepare data for plotting
    data_for_plot = {metric: [all_run_metrics[run][metric] for run in run_names] for metric in metric_names}

    # Bar chart for primary metrics
    plt.figure(figsize=(14, 7))
    x = np.arange(len(run_names))
    width = 0.15 # Bar width
    offset = np.array([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]) * width # Adjust offsets for bars

    for i, metric in enumerate(metric_names):
        plt.bar(x + offset[i], data_for_plot[metric], width, label=metric.replace('_', ' ').title())

    plt.ylabel('Score', fontsize=12)
    plt.title('Comparative Performance Metrics Across Models', fontsize=16)
    plt.xticks(x, run_names, rotation=20, ha='right', fontsize=10)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(os.path.join(os.path.dirname(save_path) if save_path else '.', 'combined_metrics_bar_chart.png'))
        print(f"Combined metrics bar chart saved to {os.path.join(os.path.dirname(save_path) if save_path else '.', 'combined_metrics_bar_chart.png')}")
    plt.show()
    plt.close()

    # Table for all metrics
    print("\n--- Comparative Metrics Table ---")
    headers = ["Metric"] + run_names
    table_data = []
    for metric in all_run_metrics[run_names[0]].keys(): # Use keys from first run to ensure all metrics are covered
        row = [metric.replace('_', ' ').title()]
        for run_name in run_names:
            value = all_run_metrics[run_name].get(metric, np.nan)
            row.append(f"{value:.4f}" if isinstance(value, (float, np.float32, np.float64)) else str(value))
        table_data.append(row)

    # Print formatted table
    col_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + table_data))]
    print(" | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * col_widths[i] for i in range(len(headers))))
    for row in table_data:
        print(" | ".join(str(item).ljust(col_widths[i]) for i, item in enumerate(row)))

# --- Main Execution Script ---
if __name__ == "__main__":
    # Instantiate base_config
    base_config = Config() 
    os.makedirs(base_config.base_log_dir, exist_ok=True) # Ensure base log directory exists

    # --- CHOOSE YOUR MODE ---
    # To run a full hyperparameter search (training each model):
    RUN_HYPERPARAMETER_SEARCH = os.environ.get("CAG_MODE", "").lower() == "search"
    # To reproduce results for a specific previously trained model:
    REPRODUCE_SPECIFIC_RUN = os.environ.get("CAG_MODE", "").lower() == "reproduce"
    # -----------------------

    if RUN_HYPERPARAMETER_SEARCH:
        # Define hyperparameter search space
        param_grid = {
            # Batch size: Keep all, as they all showed up in top runs.
            'batch_size': [8, 16, 32], # 3 values
        
            # Learning rate: Expanded range to thoroughly explore around the sweet spots.
            # Include the previously good ones and add more granular steps.
            'learning_rate': [0.005, 0.002, 0.001, 0.00075, 0.0005, 0.00025, 0.0001], # 7 values
        
            # VAE Beta: Keep the best, add intermediates, and explore a bit higher.
            # Re-added 2.0 as it appeared in top 3, and extended to 15.0 and 20.0
            # because they might offer different regularization benefits, especially with lower learning rates.
            'vae_beta': [2.0, 5.0, 7.5, 10.0, 12.5, 15.0], # 6 values
        
            # Hidden size VAE: Keep the sizes that performed well. Re-added 64 to broaden the search.
            'hidden_size_vae': [64, 128, 256, 512], # 4 values
        
            # Latent dimension: Re-added 64 to increase combinations, while keeping 128 and 256.
            'latent_dim': [64, 128, 256] # 3 values
        }
            # For quick test, you might reduce the search space:
        # param_grid = {
        #       'batch_size': [8],
        #       'learning_rate': [1e-3],
        #       'vae_beta': [1.0],
        #       'hidden_size_vae': [256],
        #       'latent_dim': [128]
        # }

        # Create all combinations of hyperparameters
        keys = param_grid.keys()
        combinations = list(itertools.product(*param_grid.values())) # Convert to list to get total count

        # Create a unique timestamped directory for the entire hyperparameter search session
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_dir = os.path.join(base_config.base_log_dir, f"search_session_{timestamp}")
        os.makedirs(session_log_dir, exist_ok=True)
        print(f"--- Starting Hyperparameter Search ({len(combinations)} combinations) in session: {session_log_dir} ---")

        # Store results for summary
        all_run_results = []

        # Iterate through each hyperparameter combination
        for i, params in enumerate(combinations):
            current_hparams = dict(zip(keys, params))
            
            # Create a new Config object for the current run
            run_config = Config()
            run_config.batch_size = current_hparams['batch_size']
            run_config.learning_rate = current_hparams['learning_rate']
            run_config.vae_beta = current_hparams['vae_beta']
            run_config.hidden_size_vae = current_hparams['hidden_size_vae']
            run_config.latent_dim = current_hparams['latent_dim']
            # The seed from base_config is already set and used by random_split

            # Generate a consistent run ID based solely on hyperparameters
            run_id = (f"bs{run_config.batch_size}_lr{run_config.learning_rate}_beta{run_config.vae_beta}"
                      f"_h{run_config.hidden_size_vae}_l{run_config.latent_dim}")
            
            # Adjust current_run_dir to be inside the session_log_dir
            current_run_dir = os.path.join(session_log_dir, run_id)

            # --- Simple Skip Logic ---
            if os.path.exists(current_run_dir) and os.path.isdir(current_run_dir): # Ensure it's a directory
                print(f"\n--- Skipping existing run: {run_id} (directory found in {session_log_dir}) ---")
                continue # Skip to the next iteration in the loop
                
            # If the run directory does not exist, proceed with the run
            run_name = run_id

            # Set the log_dir for the current run_config to the specific run directory
            run_config.log_dir = current_run_dir # IMPORTANT: Update log_dir for current run

            # Initialize TensorBoard writer for this run. It will create the directory if it doesn't exist.
            writer = SummaryWriter(log_dir=run_config.log_dir) # Use run_config.log_dir
            print(f"\n===== Starting Run: {run_name} =====")
            print(f"Hyperparameters: {current_hparams}")
            
            # --- 1. Load Data and Split for Training, Validation, Calibration, and Testing ---
            print("--- Loading Normal Data and Fitting Scaler ---")
            # Ensure a new scaler is used for each run to avoid state contamination
            run_config.data_preprocessor = MinMaxScaler()    # This line is important for scaler initialization
            # Returns: train_vae_loader, val_vae_loader, calibration_normal_dataset, test_normal_dataset
            train_vae_loader, val_vae_loader, calibration_normal_dataset, test_normal_dataset = get_normal_data_loaders(run_config)
            print(f"Number of normal samples: Train_VAE={len(train_vae_loader.dataset)}, Val_VAE={len(val_vae_loader.dataset)}, Calibration_Normal={len(calibration_normal_dataset)}, Test_Normal={len(test_normal_dataset)}")

            print("\n--- Loading Anomaly Data for Calibration and Testing ---")
            # Returns: calibration_anomaly_dataset, test_anomaly_dataset
            calibration_anomaly_dataset, test_anomaly_dataset = get_anomaly_data_loader(run_config)
            print(f"Number of anomaly samples: Calibration_Anomaly={len(calibration_anomaly_dataset)}, Test_Anomaly={len(test_anomaly_dataset)}")

            # --- 2. Initialize and Train VAE model on Normal Data ---
            vae_model = VAE(input_dim=run_config.feature_dim, hidden_size=run_config.hidden_size_vae, latent_dim=run_config.latent_dim).to(run_config.device)
            print(f"Training VAE with input_dim={vae_model.input_dim}, hidden_size={vae_model.hidden_size}, latent_dim={vae_model.latent_dim}")
            
            # Pass the writer to the training function
            best_val_loss = train_vae(vae_model, train_vae_loader, val_vae_loader, run_config, writer, run_name)    

            # --- 3. Calculate Anomaly Scores for Calibration Data ---
            print("\n--- Calculating Anomaly Scores for Calibration Data ---")
            calibration_normal_scores, calibration_normal_ids, _ = calculate_anomaly_scores_vae_only(vae_model, calibration_normal_dataset, run_config)
            calibration_anomaly_scores, calibration_anomaly_ids, _ = calculate_anomaly_scores_vae_only(vae_model, calibration_anomaly_dataset, run_config)

            # --- 4. Find Optimal Threshold using Calibration Data ---
            print("\n--- Finding Optimal Threshold using Calibration Data ---")
            optimal_threshold, calibration_metrics = find_optimal_threshold_f1(calibration_normal_scores, calibration_anomaly_scores)
            
            # Plot calibration scores with the found threshold
            # plot_anomaly_scores_split(
            #     calibration_normal_scores, 
            #     calibration_anomaly_scores, 
            #     calibration_normal_ids, # Pass the IDs here
            #     calibration_anomaly_ids, # Pass the IDs here
            #     optimal_threshold, 
            #     title=f"Calibration Anomaly Scores (Run: {run_name})"
            # )

            # --- 5. Calculate Anomaly Scores for Test Data ---
            print("\n--- Calculating Anomaly Scores for Test Data ---")
            test_normal_scores, test_normal_ids, test_normal_avg_clip_errors = calculate_anomaly_scores_vae_only(vae_model, test_normal_dataset, run_config)
            test_anomaly_scores, test_anomaly_ids, test_anomaly_avg_clip_errors = calculate_anomaly_scores_vae_only(vae_model, test_anomaly_dataset, run_config)

            # --- Save Average Clip Errors ---
            avg_clip_errors_dir = os.path.join(run_config.log_dir, "avg_clip_errors")
            os.makedirs(avg_clip_errors_dir, exist_ok=True)
            print(f"Saving average clip errors to: {avg_clip_errors_dir}")

            for idx, proc_id in enumerate(test_normal_ids):
                file_path = os.path.join(avg_clip_errors_dir, f"{proc_id}_avg_clip_error.npy")
                np.save(file_path, test_normal_avg_clip_errors[idx])
            print(f"Saved {len(test_normal_ids)} normal average clip error files.")

            for idx, proc_id in enumerate(test_anomaly_ids):
                file_path = os.path.join(avg_clip_errors_dir, f"{proc_id}_avg_clip_error.npy")
                np.save(file_path, test_anomaly_avg_clip_errors[idx])
            print(f"Saved {len(test_anomaly_ids)} anomaly average clip error files.")


            # --- 6. Evaluate Performance on Test Data using Optimal Threshold ---
            print("\n--- Evaluating Performance on Test Data ---")
            final_metrics = evaluate_performance(test_normal_scores, test_anomaly_scores, optimal_threshold)

            # Plot test scores with the found threshold
         #   plot_anomaly_scores_split(
          #      test_normal_scores, 
           #     test_anomaly_scores, 
            #    test_normal_ids, # Pass the IDs here
             #   test_anomaly_ids, # Pass the IDs here
              #  optimal_threshold, 
               # title=f"Test Anomaly Scores (Run: {run_name})"
            #)

            # --- TensorBoard Logging (Final Metrics for HPARAMS tab) ---
            writer.add_scalar('hparam/calibration_f1', calibration_metrics['f1'], 0)
            writer.add_scalar('hparam/calibration_precision', calibration_metrics['precision'], 0)
            writer.add_scalar('hparam/calibration_recall', calibration_metrics['recall'], 0)
            writer.add_scalar('hparam/optimal_threshold', optimal_threshold, 0)

            writer.add_scalar('hparam/test_overall_accuracy', final_metrics['overall_accuracy'], 0)
            writer.add_scalar('hparam/test_precision', final_metrics['overall_precision'], 0)
            writer.add_scalar('hparam/test_recall', final_metrics['overall_recall'], 0)
            writer.add_scalar('hparam/test_f1', final_metrics['overall_f1'], 0)
            writer.add_scalar('hparam/test_fpr', final_metrics['fpr'], 0)
            writer.add_scalar('hparam/test_tpr', final_metrics['tpr'], 0)
            writer.add_scalar('hparam/test_auc_roc', final_metrics['auc_roc'], 0) # Added
            writer.add_scalar('hparam/test_auc_pr', final_metrics['auc_pr'], 0)   # Added
            writer.add_scalar('hparam/test_true_positives', final_metrics['tp'], 0)
            writer.add_scalar('hparam/test_false_positives', final_metrics['fp'], 0)
            writer.add_scalar('hparam/test_true_negatives', final_metrics['tn'], 0)
            writer.add_scalar('hparam/test_false_negatives', final_metrics['fn'], 0)

            # This single call to add_hparams should now correctly link everything
            writer.add_hparams(
                hparam_dict=current_hparams,
                metric_dict={
                    'hparam/best_val_loss': best_val_loss,
                    'hparam/calibration_f1': calibration_metrics['f1'],
                    'hparam/calibration_precision': calibration_metrics['precision'],
                    'hparam/calibration_recall': calibration_metrics['recall'],
                    'hparam/optimal_threshold': optimal_threshold,
                    'hparam/test_overall_accuracy': final_metrics['overall_accuracy'],
                    'hparam/test_precision': final_metrics['overall_precision'],
                    'hparam/test_recall': final_metrics['overall_recall'],
                    'hparam/test_f1': final_metrics['overall_f1'],
                    'hparam/test_fpr': final_metrics['fpr'],
                    'hparam/test_tpr': final_metrics['tpr'],
                    'hparam/test_auc_roc': final_metrics['auc_roc'], # Added
                    'hparam/test_auc_pr': final_metrics['auc_pr'],   # Added
                    'hparam/test_true_positives': final_metrics['tp'],
                    'hparam/test_false_positives': final_metrics['fp'],
                    'hparam/test_true_negatives': final_metrics['tn'],
                    'hparam/test_false_negatives': final_metrics['fn']
                }
            )
            writer.close() # Close the writer for this run
            
            run_result = {
                'run_name': run_name,
                'hparams': current_hparams,
                'best_val_loss': best_val_loss,
                'calibration_f1': calibration_metrics['f1'],
                'optimal_threshold': optimal_threshold,
                'test_overall_accuracy': final_metrics['overall_accuracy'],
                'test_precision': final_metrics['overall_precision'],
                'test_recall': final_metrics['overall_recall'],
                'test_f1': final_metrics['overall_f1'],
                'test_auc_roc': final_metrics['auc_roc'], # Added
                'test_auc_pr': final_metrics['auc_pr']    # Added
            }
            all_run_results.append(run_result)

            print(f"===== Run {run_name} Completed =====")

        print("\n--- Hyperparameter Search Completed ---")
        print("\nSummary of all runs:")
        for result in all_run_results:
            print(f"Run: {result['run_name']}, HParams: {result['hparams']}, Test F1: {result['test_f1']:.4f}, Calib F1: {result['calibration_f1']:.4f}, Threshold: {result['optimal_threshold']:.4f}, Best Val Loss: {result['best_val_loss']:.4f}, Test AUC-ROC: {result['test_auc_roc']:.4f}, Test AUC-PR: {result['test_auc_pr']:.4f}")

        print(f"\nTo view results in TensorBoard, run: ")
        print(f"tensorboard --logdir=\"{session_log_dir}\"") # Use the session_log_dir here

    elif REPRODUCE_SPECIFIC_RUN:
        print("--- Entering Reproduce Specific Run Mode ---")

        # --- USER INPUT FOR REPRODUCTION ---
        # 1. Specify the exact timestamp of the search session
        #    (e.g., from the console output of a previous search run, or from your 'hyperparameter_search' folder)
        specific_session_timestamp = os.environ.get("CAG_SESSION", "0") # <--- REPLACE WITH YOUR SESSION TIMESTAMP

        # 2. Specify the exact hyperparameters (run_id) of the model you want to reproduce
        #    (e.g., from the run_name in the console output or folder name)
        specific_run_hparams = {
            'batch_size': 16,
            'learning_rate': 0.005,
            'vae_beta': 2.0,
            'hidden_size_vae': 512,
            'latent_dim': 256
        }
        
        # -----------------------------------

        # Reconstruct the run_id from the specified hyperparameters
        reproduce_run_id = (f"bs{specific_run_hparams['batch_size']}_lr{specific_run_hparams['learning_rate']}_beta{specific_run_hparams['vae_beta']}"
                            f"_h{specific_run_hparams['hidden_size_vae']}_l{specific_run_hparams['latent_dim']}")
        
        # Construct the full path to the specific run's directory
        reproduce_session_dir = os.path.join(base_config.base_log_dir, f"search_session_{specific_session_timestamp}")
        reproduce_run_dir = os.path.join(reproduce_session_dir, reproduce_run_id)

        # Paths to the saved model and scaler
        pretrained_model_path = os.path.join(reproduce_run_dir, "vae_model_best_run.pth")
        pretrained_scaler_path = os.path.join(reproduce_run_dir, "scaler_best_run.pkl")
        avg_clip_errors_dir = os.path.join(reproduce_run_dir, "avg_clip_errors") # Path to load saved errors

        if not os.path.exists(reproduce_run_dir):
            print(f"Error: Specified run directory not found: {reproduce_run_dir}")
            print("Please ensure the timestamp and hyperparameters match an existing run.")
        else:
            print(f"\n===== Reproducing Run: {reproduce_run_id} from session {specific_session_timestamp} =====")
            print(f"Loading model and scaler from: {reproduce_run_dir}")

            # Create a new Config object for this reproduction run, applying the specified HPs
            run_config = Config()
            run_config.batch_size = specific_run_hparams['batch_size']
            run_config.learning_rate = specific_run_hparams['learning_rate'] # Not used for inference, but kept for consistency
            run_config.vae_beta = specific_run_hparams['vae_beta']
            run_config.hidden_size_vae = specific_run_hparams['hidden_size_vae']
            run_config.latent_dim = specific_run_hparams['latent_dim']
            run_config.log_dir = reproduce_run_dir # Set log_dir for internal use (e.g., saving current plots)

            # --- 1. Load Data and ensure same splits ---
            # Data loaders will automatically use `config.seed` (which is 42) to reproduce the splits.
            print("--- Loading Normal Data for Reproduction ---")
            # Load the pre-fitted scaler first
            if os.path.exists(pretrained_scaler_path):
                with open(pretrained_scaler_path, 'rb') as f:
                    run_config.data_preprocessor = pickle.load(f)
                print(f"Successfully loaded scaler from {pretrained_scaler_path}")
            else:
                print(f"Error: Scaler not found at {pretrained_scaler_path}. Cannot reproduce results accurately.")
                exit() # Exit if scaler cannot be loaded

            train_vae_loader, val_vae_loader, calibration_normal_dataset, test_normal_dataset = get_normal_data_loaders(run_config)
            print(f"Number of normal samples: Train_VAE={len(train_vae_loader.dataset)}, Val_VAE={len(val_vae_loader.dataset)}, Calibration_Normal={len(calibration_normal_dataset)}, Test_Normal={len(test_normal_dataset)}")

            print("\n--- Loading Anomaly Data for Reproduction ---")
            calibration_anomaly_dataset, test_anomaly_dataset = get_anomaly_data_loader(run_config)
            print(f"Number of anomaly samples: Calibration_Anomaly={len(calibration_anomaly_dataset)}, Test_Anomaly={len(test_anomaly_dataset)}")

            # --- 2. Initialize VAE model and Load Pre-trained Weights ---
            vae_model = VAE(input_dim=run_config.feature_dim, hidden_size=run_config.hidden_size_vae, latent_dim=run_config.latent_dim).to(run_config.device)
            
            try:
                vae_model.load_state_dict(torch.load(pretrained_model_path, map_location=run_config.device))
                print(f"Successfully loaded model from {pretrained_model_path}")
            except Exception as e:
                print(f"Error loading model from {pretrained_model_path}: {e}")
                exit() # Exit if model loading fails

            vae_model.eval() # Set the model to evaluation mode
            print(f"VAE model initialized and loaded.")
            
            # --- 3. Calculate Anomaly Scores for Calibration Data ---
            print("\n--- Calculating Anomaly Scores for Calibration Data ---")
            calibration_normal_scores, calibration_normal_ids, _ = calculate_anomaly_scores_vae_only(vae_model, calibration_normal_dataset, run_config)
            calibration_anomaly_scores, calibration_anomaly_ids, _ = calculate_anomaly_scores_vae_only(vae_model, calibration_anomaly_dataset, run_config)

            # --- 4. Find Optimal Threshold using Calibration Data ---
            print("\n--- Finding Optimal Threshold using Calibration Data ---")
            optimal_threshold, calibration_metrics = find_optimal_threshold_f1(calibration_normal_scores, calibration_anomaly_scores)
            
            # Plot calibration scores with the found threshold
            plot_anomaly_scores_split(
                calibration_normal_scores, 
                calibration_anomaly_scores, 
                calibration_normal_ids, 
                calibration_anomaly_ids, 
                optimal_threshold, 
                title=f"Calibration Anomaly Scores (Reproduced Run: {reproduce_run_id})"
            )

            # --- 5. Calculate Anomaly Scores for Test Data ---
            print("\n--- Calculating Anomaly Scores for Test Data ---")
            # In reproduction mode, we calculate these to ensure consistency, but don't re-save them.
            test_normal_scores, test_normal_ids, test_normal_avg_clip_errors = calculate_anomaly_scores_vae_only(vae_model, test_normal_dataset, run_config)
            test_anomaly_scores, test_anomaly_ids, test_anomaly_avg_clip_errors = calculate_anomaly_scores_vae_only(vae_model, test_anomaly_dataset, run_config)
            
            # Optionally, you could add logic here to load and compare with previously saved avg_clip_errors
            # For example:
            # if os.path.exists(avg_clip_errors_dir):
            #     print(f"Loading previously saved average clip errors from {avg_clip_errors_dir} for comparison.")
            #     # Add logic to load and compare if needed

            # --- 6. Evaluate Performance on Test Data using Optimal Threshold ---
            print("\n--- Evaluating Performance on Test Data ---")
            final_metrics = evaluate_performance(test_normal_scores, test_anomaly_scores, optimal_threshold)

            # Plot test scores with the found threshold
            plot_anomaly_scores_split(
                test_normal_scores, 
                test_anomaly_scores, 
                test_normal_ids, 
                test_anomaly_ids, 
                optimal_threshold, 
                title=f"Test Anomaly Scores (Reproduced Run: {reproduce_run_id})"
            )
            print(f"\n--- Reproduction of Run {reproduce_run_id} Completed ---")

    else:
        print("Please set RUN_HYPERPARAMETER_SEARCH or REPRODUCE_SPECIFIC_RUN to True in the main block.")

# --- Main Execution Script ---
if __name__ == "__main__2":
    # Instantiate base_config
    base_config = Config() 
    os.makedirs(base_config.base_log_dir, exist_ok=True) # Ensure base log directory exists

    # --- CHOOSE YOUR MODE ---
    # To run a full hyperparameter search (training each model):
    RUN_HYPERPARAMETER_SEARCH = True
    # To reproduce results for a specific previously trained model:
    REPRODUCE_SPECIFIC_RUN = False 
    # -----------------------

    if RUN_HYPERPARAMETER_SEARCH:
        # Define hyperparameter search space
        param_grid = {
        # Batch size: Narrow around the best performing values (8 to 32).
        'batch_size': [8, 16, 32],  
    
        # Learning rate: Focus on the successful range.
        # 0.01 yielded some best F1s, 0.0001 also good. Let's explore values in between and around.
        'learning_rate': [0.005, 0.001, 0.0005, 0.0001],  
    
        # VAE Beta: Crucially, explore higher values, as 5.0 and 10.0 yielded best F1.
        # Extending to 15.0 to see if even more regularization is beneficial.
        'vae_beta': [2.0, 5.0, 7.5, 10.0, 15.0],  
    
        # Hidden size VAE: Keep the sizes that performed well (128, 256, 512).
        # Consider if your VAE supports multi-layer configurations and adjust if needed.
        'hidden_size_vae': [64,128, 256, 512],  
    
        # Latent dimension: Focus around 256 and 512, which appeared in the top F1 runs.
        # Added 768 to explore a slightly larger option.
        'latent_dim': [64,128, 256]
        }
            # For quick test, you might reduce the search space:
        # param_grid = {
        #       'batch_size': [8],
        #       'learning_rate': [1e-3],
        #       'vae_beta': [1.0],
        #       'hidden_size_vae': [256],
        #       'latent_dim': [128]
        # }

        # Create all combinations of hyperparameters
        keys = param_grid.keys()
        combinations = list(itertools.product(*param_grid.values())) # Convert to list to get total count

        # Create a unique timestamped directory for the entire hyperparameter search session
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_dir = os.path.join(base_config.base_log_dir, f"search_session_{timestamp}")
        os.makedirs(session_log_dir, exist_ok=True)
        print(f"--- Starting Hyperparameter Search ({len(combinations)} combinations) in session: {session_log_dir} ---")

        # Store results for summary
        all_run_results = []

        # Iterate through each hyperparameter combination
        for i, params in enumerate(combinations):
            current_hparams = dict(zip(keys, params))
            
            # Create a new Config object for the current run
            run_config = Config()
            run_config.batch_size = current_hparams['batch_size']
            run_config.learning_rate = current_hparams['learning_rate']
            run_config.vae_beta = current_hparams['vae_beta']
            run_config.hidden_size_vae = current_hparams['hidden_size_vae']
            run_config.latent_dim = current_hparams['latent_dim']
            # The seed from base_config is already set and used by random_split

            # Generate a consistent run ID based solely on hyperparameters
            run_id = (f"bs{run_config.batch_size}_lr{run_config.learning_rate}_beta{run_config.vae_beta}"
                      f"_h{run_config.hidden_size_vae}_l{run_config.latent_dim}")
            
            # Adjust current_run_dir to be inside the session_log_dir
            current_run_dir = os.path.join(session_log_dir, run_id)

            # --- Simple Skip Logic ---
            if os.path.exists(current_run_dir) and os.path.isdir(current_run_dir): # Ensure it's a directory
                print(f"\n--- Skipping existing run: {run_id} (directory found in {session_log_dir}) ---")
                continue # Skip to the next iteration in the loop
                
            # If the run directory does not exist, proceed with the run
            run_name = run_id

            # Set the log_dir for the current run_config to the specific run directory
            run_config.log_dir = current_run_dir # IMPORTANT: Update log_dir for current run

            # Initialize TensorBoard writer for this run. It will create the directory if it doesn't exist.
            writer = SummaryWriter(log_dir=run_config.log_dir) # Use run_config.log_dir
            print(f"\n===== Starting Run: {run_name} =====")
            print(f"Hyperparameters: {current_hparams}")
            
            # --- 1. Load Data and Split for Training, Validation, Calibration, and Testing ---
            print("--- Loading Normal Data and Fitting Scaler ---")
            # Ensure a new scaler is used for each run to avoid state contamination
            run_config.data_preprocessor = MinMaxScaler()    # This line is important for scaler initialization
            # Returns: train_vae_loader, val_vae_loader, calibration_normal_dataset, test_normal_dataset
            train_vae_loader, val_vae_loader, calibration_normal_dataset, test_normal_dataset = get_normal_data_loaders(run_config)
            print(f"Number of normal samples: Train_VAE={len(train_vae_loader.dataset)}, Val_VAE={len(val_vae_loader.dataset)}, Calibration_Normal={len(calibration_normal_dataset)}, Test_Normal={len(test_normal_dataset)}")

            print("\n--- Loading Anomaly Data for Calibration and Testing ---")
            # Returns: calibration_anomaly_dataset, test_anomaly_dataset
            calibration_anomaly_dataset, test_anomaly_dataset = get_anomaly_data_loader(run_config)
            print(f"Number of anomaly samples: Calibration_Anomaly={len(calibration_anomaly_dataset)}, Test_Anomaly={len(test_anomaly_dataset)}")

            # --- 2. Initialize and Train VAE model on Normal Data ---
            vae_model = VAE(input_dim=run_config.feature_dim, hidden_size=run_config.hidden_size_vae, latent_dim=run_config.latent_dim).to(run_config.device)
            print(f"Training VAE with input_dim={vae_model.input_dim}, hidden_size={vae_model.hidden_size}, latent_dim={vae_model.latent_dim}")
            
            # Pass the writer to the training function
            best_val_loss = train_vae(vae_model, train_vae_loader, val_vae_loader, run_config, writer, run_name)    

            # --- 3. Calculate Anomaly Scores for Calibration Data ---
            print("\n--- Calculating Anomaly Scores for Calibration Data ---")
            calibration_normal_scores, calibration_normal_ids = calculate_anomaly_scores_vae_only(vae_model, calibration_normal_dataset, run_config)
            calibration_anomaly_scores, calibration_anomaly_ids = calculate_anomaly_scores_vae_only(vae_model, calibration_anomaly_dataset, run_config)

            # --- 4. Find Optimal Threshold using Calibration Data ---
            print("\n--- Finding Optimal Threshold using Calibration Data ---")
            optimal_threshold, calibration_metrics = find_optimal_threshold_f1(calibration_normal_scores, calibration_anomaly_scores)
            
            # Plot calibration scores with the found threshold
           # plot_anomaly_scores_split(
            #    calibration_normal_scores, 
             #   calibration_anomaly_scores, 
              #  calibration_normal_ids, # Pass the IDs here
               # calibration_anomaly_ids, # Pass the IDs here
               # optimal_threshold, 
               # title=f"Calibration Anomaly Scores (Run: {run_name})"
            #)

            # --- 5. Calculate Anomaly Scores for Test Data ---
            print("\n--- Calculating Anomaly Scores for Test Data ---")
            test_normal_scores, test_normal_ids = calculate_anomaly_scores_vae_only(vae_model, test_normal_dataset, run_config)
            test_anomaly_scores, test_anomaly_ids = calculate_anomaly_scores_vae_only(vae_model, test_anomaly_dataset, run_config)

            # --- 6. Evaluate Performance on Test Data using Optimal Threshold ---
            print("\n--- Evaluating Performance on Test Data ---")
            final_metrics = evaluate_performance(test_normal_scores, test_anomaly_scores, optimal_threshold)

            # Plot test scores with the found threshold
            #plot_anomaly_scores_split(
             #   test_normal_scores, 
              #  test_anomaly_scores, 
               # test_normal_ids, # Pass the IDs here
                #test_anomaly_ids, # Pass the IDs here
                #optimal_threshold, 
                #title=f"Test Anomaly Scores (Run: {run_name})"
            #)

            # --- TensorBoard Logging (Final Metrics for HPARAMS tab) ---
            writer.add_scalar('hparam/calibration_f1', calibration_metrics['f1'], 0)
            writer.add_scalar('hparam/calibration_precision', calibration_metrics['precision'], 0)
            writer.add_scalar('hparam/calibration_recall', calibration_metrics['recall'], 0)
            writer.add_scalar('hparam/optimal_threshold', optimal_threshold, 0)

            writer.add_scalar('hparam/test_overall_accuracy', final_metrics['overall_accuracy'], 0)
            writer.add_scalar('hparam/test_precision', final_metrics['overall_precision'], 0)
            writer.add_scalar('hparam/test_recall', final_metrics['overall_recall'], 0)
            writer.add_scalar('hparam/test_f1', final_metrics['overall_f1'], 0)
            writer.add_scalar('hparam/test_fpr', final_metrics['fpr'], 0)
            writer.add_scalar('hparam/test_tpr', final_metrics['tpr'], 0)
         #   writer.add_scalar('hparam/test_auc_roc', final_metrics['auc_roc'], 0) # Added
          #  writer.add_scalar('hparam/test_auc_pr', final_metrics['auc_pr'], 0)   # Added
            writer.add_scalar('hparam/test_true_positives', final_metrics['tp'], 0)
            writer.add_scalar('hparam/test_false_positives', final_metrics['fp'], 0)
            writer.add_scalar('hparam/test_true_negatives', final_metrics['tn'], 0)
            writer.add_scalar('hparam/test_false_negatives', final_metrics['fn'], 0)

            # This single call to add_hparams should now correctly link everything
            writer.add_hparams(
                hparam_dict=current_hparams,
                metric_dict={
                    'hparam/best_val_loss': best_val_loss,
                    'hparam/calibration_f1': calibration_metrics['f1'],
                    'hparam/calibration_precision': calibration_metrics['precision'],
                    'hparam/calibration_recall': calibration_metrics['recall'],
                    'hparam/optimal_threshold': optimal_threshold,
                    'hparam/test_overall_accuracy': final_metrics['overall_accuracy'],
                    'hparam/test_precision': final_metrics['overall_precision'],
                    'hparam/test_recall': final_metrics['overall_recall'],
                    'hparam/test_f1': final_metrics['overall_f1'],
                    'hparam/test_fpr': final_metrics['fpr'],
                    'hparam/test_tpr': final_metrics['tpr'],
           #         'hparam/test_auc_roc': final_metrics['auc_roc'], # Added
            #        'hparam/test_auc_pr': final_metrics['auc_pr'],   # Added
                    'hparam/test_true_positives': final_metrics['tp'],
                    'hparam/test_false_positives': final_metrics['fp'],
                    'hparam/test_true_negatives': final_metrics['tn'],
                    'hparam/test_false_negatives': final_metrics['fn']
                }
            )
            writer.close() # Close the writer for this run
            
            run_result = {
                'run_name': run_name,
                'hparams': current_hparams,
                'best_val_loss': best_val_loss,
                'calibration_f1': calibration_metrics['f1'],
                'optimal_threshold': optimal_threshold,
                'test_overall_accuracy': final_metrics['overall_accuracy'],
                'test_precision': final_metrics['overall_precision'],
                'test_recall': final_metrics['overall_recall'],
                'test_f1': final_metrics['overall_f1'],
              #  'test_auc_roc': final_metrics['auc_roc'], # Added
             #   'test_auc_pr': final_metrics['auc_pr']    # Added
            }
            all_run_results.append(run_result)

            print(f"===== Run {run_name} Completed =====")

        print("\n--- Hyperparameter Search Completed ---")
        print("\nSummary of all runs:")
        for result in all_run_results:
            print(f"Run: {result['run_name']}, HParams: {result['hparams']}, Test F1: {result['test_f1']:.4f}, Calib F1: {result['calibration_f1']:.4f}, Threshold: {result['optimal_threshold']:.4f}, Best Val Loss: {result['best_val_loss']:.4f}")

        print(f"\nTo view results in TensorBoard, run: ")
        print(f"tensorboard --logdir=\"{session_log_dir}\"") # Use the session_log_dir here

    elif REPRODUCE_SPECIFIC_RUN:
        print("--- Entering Reproduce Specific Run Mode ---")

        # --- USER INPUT FOR REPRODUCTION ---
        # 1. Specify the exact timestamp of the search session
        #    (e.g., from the console output of a previous search run, or from your 'hyperparameter_search' folder)
        specific_session_timestamp = "20250617_163035" # <--- REPLACE WITH YOUR SESSION TIMESTAMP

        # 2. Specify the exact hyperparameters (run_id) of the model you want to reproduce
        #    (e.g., from the run_name in the console output or folder name)
        specific_run_hparams = {
            'batch_size': 16,
            'learning_rate': 5e-4,
            'vae_beta': 10,
            'hidden_size_vae': 128,
            'latent_dim': 64
        }
        # -----------------------------------

        # Reconstruct the run_id from the specified hyperparameters
        reproduce_run_id = (f"bs{specific_run_hparams['batch_size']}_lr{specific_run_hparams['learning_rate']}_beta{specific_run_hparams['vae_beta']}"
                            f"_h{specific_run_hparams['hidden_size_vae']}_l{specific_run_hparams['latent_dim']}")
        
        # Construct the full path to the specific run's directory
        reproduce_session_dir = os.path.join(base_config.base_log_dir, f"search_session_{specific_session_timestamp}")
        reproduce_run_dir = os.path.join(reproduce_session_dir, reproduce_run_id)

        # Paths to the saved model and scaler
        pretrained_model_path = os.path.join(reproduce_run_dir, "vae_model_best_run.pth")
        pretrained_scaler_path = os.path.join(reproduce_run_dir, "scaler_best_run.pkl")

        if not os.path.exists(reproduce_run_dir):
            print(f"Error: Specified run directory not found: {reproduce_run_dir}")
            print("Please ensure the timestamp and hyperparameters match an existing run.")
        else:
            print(f"\n===== Reproducing Run: {reproduce_run_id} from session {specific_session_timestamp} =====")
            print(f"Loading model and scaler from: {reproduce_run_dir}")

            # Create a new Config object for this reproduction run, applying the specified HPs
            run_config = Config()
            run_config.batch_size = specific_run_hparams['batch_size']
            run_config.learning_rate = specific_run_hparams['learning_rate'] # Not used for inference, but kept for consistency
            run_config.vae_beta = specific_run_hparams['vae_beta']
            run_config.hidden_size_vae = specific_run_hparams['hidden_size_vae']
            run_config.latent_dim = specific_run_hparams['latent_dim']
            run_config.log_dir = reproduce_run_dir # Set log_dir for internal use (e.g., saving current plots)

            # --- 1. Load Data and ensure same splits ---
            # Data loaders will automatically use `config.seed` (which is 42) to reproduce the splits.
            print("--- Loading Normal Data for Reproduction ---")
            # Load the pre-fitted scaler first
            if os.path.exists(pretrained_scaler_path):
                with open(pretrained_scaler_path, 'rb') as f:
                    run_config.data_preprocessor = pickle.load(f)
                print(f"Successfully loaded scaler from {pretrained_scaler_path}")
            else:
                print(f"Error: Scaler not found at {pretrained_scaler_path}. Cannot reproduce results accurately.")
                exit() # Exit if scaler cannot be loaded

            train_vae_loader, val_vae_loader, calibration_normal_dataset, test_normal_dataset = get_normal_data_loaders(run_config)
            print(f"Number of normal samples: Train_VAE={len(train_vae_loader.dataset)}, Val_VAE={len(val_vae_loader.dataset)}, Calibration_Normal={len(calibration_normal_dataset)}, Test_Normal={len(test_normal_dataset)}")

            print("\n--- Loading Anomaly Data for Reproduction ---")
            calibration_anomaly_dataset, test_anomaly_dataset = get_anomaly_data_loader(run_config)
            print(f"Number of anomaly samples: Calibration_Anomaly={len(calibration_anomaly_dataset)}, Test_Anomaly={len(test_anomaly_dataset)}")

            # --- 2. Initialize VAE model and Load Pre-trained Weights ---
            vae_model = VAE(input_dim=run_config.feature_dim, hidden_size=run_config.hidden_size_vae, latent_dim=run_config.latent_dim).to(run_config.device)
            
            try:
                vae_model.load_state_dict(torch.load(pretrained_model_path, map_location=run_config.device))
                print(f"Successfully loaded model from {pretrained_model_path}")
            except Exception as e:
                print(f"Error loading model from {pretrained_model_path}: {e}")
                exit() # Exit if model loading fails

            vae_model.eval() # Set the model to evaluation mode
            print(f"VAE model initialized and loaded.")
            
            # --- 3. Calculate Anomaly Scores for Calibration Data ---
            print("\n--- Calculating Anomaly Scores for Calibration Data ---")
            calibration_normal_scores, calibration_normal_ids = calculate_anomaly_scores_vae_only(vae_model, calibration_normal_dataset, run_config)
            calibration_anomaly_scores, calibration_anomaly_ids = calculate_anomaly_scores_vae_only(vae_model, calibration_anomaly_dataset, run_config)

            # --- 4. Find Optimal Threshold using Calibration Data ---
            print("\n--- Finding Optimal Threshold using Calibration Data ---")
            optimal_threshold, calibration_metrics = find_optimal_threshold_f1(calibration_normal_scores, calibration_anomaly_scores)
            
            # Plot calibration scores with the found threshold
            plot_anomaly_scores_split(
                calibration_normal_scores, 
                calibration_anomaly_scores, 
                calibration_normal_ids, 
                calibration_anomaly_ids, 
                optimal_threshold, 
                title=f"Calibration Anomaly Scores (Reproduced Run: {reproduce_run_id})"
            )

            # --- 5. Calculate Anomaly Scores for Test Data ---
            print("\n--- Calculating Anomaly Scores for Test Data ---")
            test_normal_scores, test_normal_ids = calculate_anomaly_scores_vae_only(vae_model, test_normal_dataset, run_config)
            test_anomaly_scores, test_anomaly_ids = calculate_anomaly_scores_vae_only(vae_model, test_anomaly_dataset, run_config)

            # --- 6. Evaluate Performance on Test Data using Optimal Threshold ---
            print("\n--- Evaluating Performance on Test Data ---")
            final_metrics = evaluate_performance(test_normal_scores, test_anomaly_scores, optimal_threshold)

            # Plot test scores with the found threshold
            plot_anomaly_scores_split(
                test_normal_scores, 
                test_anomaly_scores, 
                test_normal_ids, 
                test_anomaly_ids, 
                optimal_threshold, 
                title=f"Test Anomaly Scores (Reproduced Run: {reproduce_run_id})"
            )
            print(f"\n--- Reproduction of Run {reproduce_run_id} Completed ---")

    else:
        print("Please set RUN_HYPERPARAMETER_SEARCH or REPRODUCE_SPECIFIC_RUN to True in the main block.")
