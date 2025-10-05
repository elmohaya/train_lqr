"""
Training Script for Universal LQR Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    SEQUENCE_LENGTH, TRANSFORMER_CONFIG, TRAINING_CONFIG,
    NORMALIZATION_STRATEGY, DATA_DIR, MODEL_DIR, LOG_DIR, RANDOM_SEED,
    MAX_STATE_DIM, MAX_INPUT_DIM, DIMENSION_ENCODING_SIZE
)
from transformer_model import UniversalLQRTransformer, count_parameters
from data_generation import load_data
from data_utils import create_dimension_encoding, create_control_mask


class LQRDataset(Dataset):
    """
    PyTorch Dataset for LQR training data.
    """
    
    def __init__(self, data, sequence_length, normalization='standardize', 
                 state_mean=None, state_std=None, control_mean=None, control_std=None,
                 max_state_dim=None, max_control_dim=None):
        """
        Args:
            data: List of dataset dictionaries
            sequence_length: Length of state sequence to use
            normalization: Normalization strategy ('standardize', 'minmax', 'none')
            state_mean, state_std: Precomputed statistics (for validation/test)
            control_mean, control_std: Precomputed statistics
            max_state_dim: Maximum state dimension (for padding)
            max_control_dim: Maximum control dimension (for padding)
        """
        self.sequence_length = sequence_length
        self.normalization = normalization
        
        # Extract all trajectories
        self.samples = []
        all_states = []
        all_controls = []
        
        # Determine max dimensions
        if max_state_dim is None or max_control_dim is None:
            max_state_dim = 0
            max_control_dim = 0
            for system_data in data:
                max_state_dim = max(max_state_dim, system_data['n_states'])
                max_control_dim = max(max_control_dim, system_data['n_inputs'])
        
        self.max_state_dim = max_state_dim
        self.max_control_dim = max_control_dim
        
        # Extract sequences from all trajectories
        for system_data in data:
            n_states = system_data['n_states']
            n_inputs = system_data['n_inputs']
            
            for trajectory in system_data['trajectories']:
                states = trajectory['states']  # (T, n_states)
                controls = trajectory['controls']  # (T, n_inputs)
                
                # Create sequences of length sequence_length
                T = len(states)
                for i in range(sequence_length, T):
                    state_seq = states[i-sequence_length:i]  # (seq_len, n_states)
                    control_target = controls[i]  # (n_inputs,)
                    
                    self.samples.append({
                        'states': state_seq,
                        'control': control_target,
                        'n_states': n_states,
                        'n_inputs': n_inputs
                    })
                    
                    all_states.append(state_seq)
                    all_controls.append(control_target)
        
        # Compute normalization statistics
        if normalization == 'standardize':
            if state_mean is None:
                # Compute statistics from data
                # Flatten all states and controls
                all_states_concat = []
                all_controls_concat = []
                
                for sample in self.samples:
                    states = sample['states']  # (seq_len, n_states)
                    control = sample['control']  # (n_inputs,)
                    n_states = sample['n_states']
                    n_inputs = sample['n_inputs']
                    
                    all_states_concat.append(states.flatten()[:n_states * self.sequence_length])
                    all_controls_concat.append(control[:n_inputs])
                
                # For simplicity, compute per-dimension statistics on max dimensions
                # This is approximate but works for system-agnostic learning
                self.state_mean = np.zeros(max_state_dim)
                self.state_std = np.ones(max_state_dim)
                self.control_mean = np.zeros(max_control_dim)
                self.control_std = np.ones(max_control_dim)
                
                # Compute actual statistics
                for dim in range(max_state_dim):
                    dim_values = []
                    for sample in self.samples:
                        if dim < sample['n_states']:
                            dim_values.extend(sample['states'][:, dim])
                    if len(dim_values) > 0:
                        self.state_mean[dim] = np.mean(dim_values)
                        self.state_std[dim] = np.std(dim_values) + 1e-8
                
                for dim in range(max_control_dim):
                    dim_values = []
                    for sample in self.samples:
                        if dim < sample['n_inputs']:
                            dim_values.append(sample['control'][dim])
                    if len(dim_values) > 0:
                        self.control_mean[dim] = np.mean(dim_values)
                        self.control_std[dim] = np.std(dim_values) + 1e-8
            else:
                self.state_mean = state_mean
                self.state_std = state_std
                self.control_mean = control_mean
                self.control_std = control_std
        else:
            self.state_mean = np.zeros(max_state_dim)
            self.state_std = np.ones(max_state_dim)
            self.control_mean = np.zeros(max_control_dim)
            self.control_std = np.ones(max_control_dim)
        
        print(f"Dataset created: {len(self.samples)} samples")
        print(f"  Max state dim: {self.max_state_dim}")
        print(f"  Max control dim: {self.max_control_dim}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        states = sample['states'].copy()  # (seq_len, n_states)
        control = sample['control'].copy()  # (n_inputs,)
        n_states = sample['n_states']
        n_inputs = sample['n_inputs']
        
        # Normalize
        if self.normalization == 'standardize':
            states[:, :n_states] = (states[:, :n_states] - self.state_mean[:n_states]) / self.state_std[:n_states]
            control[:n_inputs] = (control[:n_inputs] - self.control_mean[:n_inputs]) / self.control_std[:n_inputs]
        
        # Pad to max dimensions
        states_padded = np.zeros((self.sequence_length, self.max_state_dim))
        states_padded[:, :n_states] = states
        
        control_padded = np.zeros(self.max_control_dim)
        control_padded[:n_inputs] = control
        
        # Create dimension encoding (binary representation)
        dim_encoding = create_dimension_encoding(n_inputs, n_states, device='cpu').numpy()
        
        # Concatenate dimension encoding to each timestep of states
        # states_padded: (seq_len, max_state_dim)
        # dim_encoding: (encoding_size,)
        # Replicate encoding for each timestep
        dim_encoding_repeated = np.tile(dim_encoding, (self.sequence_length, 1))  # (seq_len, encoding_size)
        input_sequence = np.concatenate([states_padded, dim_encoding_repeated], axis=1)  # (seq_len, max_state_dim + encoding_size)
        
        # Create control mask (1 for active dims, 0 for padded)
        control_mask = np.zeros(self.max_control_dim)
        control_mask[:n_inputs] = 1.0
        
        return {
            'input_sequence': torch.FloatTensor(input_sequence),
            'control': torch.FloatTensor(control_padded),
            'control_mask': torch.FloatTensor(control_mask),
            'n_states': n_states,
            'n_inputs': n_inputs
        }
    
    def get_statistics(self):
        """Return normalization statistics."""
        return {
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'control_mean': self.control_mean,
            'control_std': self.control_std,
            'max_state_dim': self.max_state_dim,
            'max_control_dim': self.max_control_dim
        }


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    """
    Train for one epoch with masked loss.
    """
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_sequence = batch['input_sequence'].to(device)  # (batch, seq_len, max_state_dim + encoding_size)
        controls_target = batch['control'].to(device)  # (batch, max_control_dim)
        control_mask = batch['control_mask'].to(device)  # (batch, max_control_dim)
        
        # Forward pass
        controls_pred = model(input_sequence, control_mask=control_mask)  # (batch, seq_len, max_control_dim)
        
        # We only care about the prediction at the last time step
        controls_pred_last = controls_pred[:, -1, :]  # (batch, max_control_dim)
        
        # Compute masked loss (only on active control dimensions)
        loss_per_dim = (controls_pred_last - controls_target) ** 2  # (batch, max_control_dim)
        masked_loss = loss_per_dim * control_mask  # Zero out padded dimensions
        loss = masked_loss.sum() / control_mask.sum()  # Average over active dimensions only
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Validate the model with masked loss.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_sequence = batch['input_sequence'].to(device)
            controls_target = batch['control'].to(device)
            control_mask = batch['control_mask'].to(device)
            
            controls_pred = model(input_sequence, control_mask=control_mask)
            controls_pred_last = controls_pred[:, -1, :]
            
            # Compute masked loss
            loss_per_dim = (controls_pred_last - controls_target) ** 2
            masked_loss = loss_per_dim * control_mask
            loss = masked_loss.sum() / control_mask.sum()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_model(model, train_loader, val_loader, config, device, save_dir=MODEL_DIR):
    """
    Train the transformer model.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer with learning rate warmup
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                            betas=(0.9, 0.999), weight_decay=0.01)
    
    # Learning rate scheduler (cosine annealing with warmup)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                                config['gradient_clip'])
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  [OK] Best model saved!")
        
        # Save checkpoint periodically
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Save final model
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, os.path.join(save_dir, 'final_model.pt'))
    
    return history


def plot_training_history(history, save_dir=LOG_DIR):
    """Plot training history."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


def main():
    """Main training function."""
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    all_data = load_data(DATA_DIR)
    
    if all_data is None:
        print("No data found! Please run data_generation.py first.")
        return
    
    # Split into train and validation
    val_split_idx = int(len(all_data) * (1 - TRAINING_CONFIG['validation_split']))
    train_data = all_data[:val_split_idx]
    val_data = all_data[val_split_idx:]
    
    print(f"Train systems: {len(train_data)}")
    print(f"Val systems: {len(val_data)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = LQRDataset(train_data, SEQUENCE_LENGTH, NORMALIZATION_STRATEGY)
    
    # Use same statistics for validation
    stats = train_dataset.get_statistics()
    val_dataset = LQRDataset(val_data, SEQUENCE_LENGTH, NORMALIZATION_STRATEGY,
                            state_mean=stats['state_mean'],
                            state_std=stats['state_std'],
                            control_mean=stats['control_mean'],
                            control_std=stats['control_std'],
                            max_state_dim=stats['max_state_dim'],
                            max_control_dim=stats['max_control_dim'])
    
    # Save statistics
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'normalization_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    # Create dataloaders (reduce num_workers to save memory)
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'],
                             shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'],
                           shuffle=False, num_workers=2, pin_memory=False)
    
    # Create model
    print("\nCreating model...")
    model = UniversalLQRTransformer(
        max_state_dim=stats['max_state_dim'],
        max_control_dim=stats['max_control_dim'],
        dimension_encoding_size=DIMENSION_ENCODING_SIZE,
        d_model=TRANSFORMER_CONFIG['d_model'],
        n_heads=TRANSFORMER_CONFIG['n_heads'],
        n_layers=TRANSFORMER_CONFIG['n_layers'],
        d_ff=TRANSFORMER_CONFIG['d_ff'],
        dropout=TRANSFORMER_CONFIG['dropout'],
        max_seq_len=TRANSFORMER_CONFIG['max_seq_len']
    )
    model = model.to(device)
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Train
    history = train_model(model, train_loader, val_loader, TRAINING_CONFIG, device)
    
    # Plot training history
    plot_training_history(history)
    
    print("\nTraining complete!")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Logs saved to: {LOG_DIR}")


if __name__ == '__main__':
    main()

