"""
Memory-Efficient Training Script for Universal LQR Transformer

Uses on-the-fly sequence generation instead of pre-loading everything.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import random as py_random

from config import (
    SEQUENCE_LENGTH, TRANSFORMER_CONFIG, TRAINING_CONFIG,
    NORMALIZATION_STRATEGY, DATA_DIR, MODEL_DIR, LOG_DIR, RANDOM_SEED,
    MAX_STATE_DIM, MAX_INPUT_DIM, DIMENSION_ENCODING_SIZE
)
from transformer_model import UniversalLQRTransformer, count_parameters
from data_generation import load_data
from data_utils import create_dimension_encoding, create_control_mask


class MemoryEfficientLQRDataset(Dataset):
    """
    Memory-efficient dataset that generates sequences on-the-fly.
    Stores only file paths and metadata, not the actual trajectory data.
    """
    
    def __init__(self, data_file, sequence_length, normalization='standardize',
                 max_trajectories_per_system=None, subset_ratio=1.0):
        """
        Args:
            data_file: Path to pickled data
            sequence_length: Length of sequences
            normalization: Normalization strategy
            max_trajectories_per_system: Limit trajectories per system (for memory)
            subset_ratio: Use only a fraction of data (e.g., 0.1 for 10%)
        """
        self.sequence_length = sequence_length
        self.normalization = normalization
        self.max_state_dim = MAX_STATE_DIM
        self.max_control_dim = MAX_INPUT_DIM
        
        print(f"Loading data metadata from {data_file}...")
        with open(data_file, 'rb') as f:
            self.all_data = pickle.load(f)
        
        print(f"Loaded {len(self.all_data)} system variants")
        
        # Subsample if requested
        if subset_ratio < 1.0:
            n_keep = max(1, int(len(self.all_data) * subset_ratio))
            self.all_data = self.all_data[:n_keep]
            print(f"Using {len(self.all_data)} system variants ({subset_ratio*100}% of data)")
        
        # Build index of all valid sequences
        self.sequence_index = []
        total_sequences = 0
        
        for sys_idx, system_data in enumerate(self.all_data):
            n_trajectories = len(system_data['trajectories'])
            if max_trajectories_per_system:
                n_trajectories = min(n_trajectories, max_trajectories_per_system)
            
            for traj_idx in range(n_trajectories):
                traj_len = len(system_data['trajectories'][traj_idx]['time'])
                n_sequences = traj_len - sequence_length
                
                if n_sequences > 0:
                    for seq_start in range(n_sequences):
                        self.sequence_index.append((sys_idx, traj_idx, seq_start))
                        total_sequences += 1
        
        print(f"Total sequences available: {total_sequences:,}")
        
        # Compute normalization statistics (approximate, on subset)
        self.compute_normalization_stats()
    
    def compute_normalization_stats(self):
        """Compute normalization statistics on a sample of data."""
        print("Computing normalization statistics...")
        
        # Sample sequences for statistics
        n_samples = min(10000, len(self.sequence_index))
        sample_indices = np.random.choice(len(self.sequence_index), n_samples, replace=False)
        
        state_values = [[] for _ in range(self.max_state_dim)]
        control_values = [[] for _ in range(self.max_control_dim)]
        
        for idx in tqdm(sample_indices, desc="Sampling for stats"):
            sys_idx, traj_idx, seq_start = self.sequence_index[idx]
            system_data = self.all_data[sys_idx]
            trajectory = system_data['trajectories'][traj_idx]
            
            states = trajectory['states'][seq_start:seq_start + self.sequence_length]
            control = trajectory['controls'][seq_start + self.sequence_length]
            
            n_states = system_data['n_states']
            n_inputs = system_data['n_inputs']
            
            for dim in range(n_states):
                state_values[dim].extend(states[:, dim])
            for dim in range(n_inputs):
                control_values[dim].append(control[dim])
        
        # Compute mean and std
        self.state_mean = np.zeros(self.max_state_dim)
        self.state_std = np.ones(self.max_state_dim)
        self.control_mean = np.zeros(self.max_control_dim)
        self.control_std = np.ones(self.max_control_dim)
        
        if self.normalization == 'standardize':
            for dim in range(self.max_state_dim):
                if len(state_values[dim]) > 0:
                    self.state_mean[dim] = np.mean(state_values[dim])
                    self.state_std[dim] = np.std(state_values[dim]) + 1e-8
            
            for dim in range(self.max_control_dim):
                if len(control_values[dim]) > 0:
                    self.control_mean[dim] = np.mean(control_values[dim])
                    self.control_std[dim] = np.std(control_values[dim]) + 1e-8
        
        print("Statistics computed.")
    
    def __len__(self):
        return len(self.sequence_index)
    
    def __getitem__(self, idx):
        sys_idx, traj_idx, seq_start = self.sequence_index[idx]
        system_data = self.all_data[sys_idx]
        trajectory = system_data['trajectories'][traj_idx]
        
        # Extract sequence
        states = trajectory['states'][seq_start:seq_start + self.sequence_length].copy()
        control = trajectory['controls'][seq_start + self.sequence_length].copy()
        
        n_states = system_data['n_states']
        n_inputs = system_data['n_inputs']
        
        # Normalize
        if self.normalization == 'standardize':
            states[:, :n_states] = (states[:, :n_states] - self.state_mean[:n_states]) / self.state_std[:n_states]
            control[:n_inputs] = (control[:n_inputs] - self.control_mean[:n_inputs]) / self.control_std[:n_inputs]
        
        # Pad to max dimensions
        states_padded = np.zeros((self.sequence_length, self.max_state_dim))
        states_padded[:, :n_states] = states
        
        control_padded = np.zeros(self.max_control_dim)
        control_padded[:n_inputs] = control
        
        # Create dimension encoding (already returns numpy array)
        dim_encoding = create_dimension_encoding(n_inputs, n_states)
        # Convert torch tensor to numpy if needed
        if hasattr(dim_encoding, 'numpy'):
            dim_encoding = dim_encoding.numpy()
        dim_encoding_repeated = np.tile(dim_encoding, (self.sequence_length, 1))
        input_sequence = np.concatenate([states_padded, dim_encoding_repeated], axis=1)
        
        # Create control mask
        control_mask = np.zeros(self.max_control_dim)
        control_mask[:n_inputs] = 1.0
        
        return {
            'input_sequence': torch.FloatTensor(input_sequence),
            'control': torch.FloatTensor(control_padded),
            'control_mask': torch.FloatTensor(control_mask),
            'n_states': n_states,
            'n_inputs': n_inputs
        }


def train_epoch(model, dataloader, optimizer, device, use_amp=True):
    """Train for one epoch with masked loss and mixed precision."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    # Use automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    for batch in tqdm(dataloader, desc="Training"):
        input_sequence = batch['input_sequence'].to(device, non_blocking=True)
        controls_target = batch['control'].to(device, non_blocking=True)
        control_mask = batch['control_mask'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                controls_pred = model(input_sequence, control_mask=control_mask)
                controls_pred_last = controls_pred[:, -1, :]
                
                # Compute masked loss
                loss_per_dim = (controls_pred_last - controls_target) ** 2
                masked_loss = loss_per_dim * control_mask
                loss = masked_loss.sum() / control_mask.sum()
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            controls_pred = model(input_sequence, control_mask=control_mask)
            controls_pred_last = controls_pred[:, -1, :]
            
            loss_per_dim = (controls_pred_last - controls_target) ** 2
            masked_loss = loss_per_dim * control_mask
            loss = masked_loss.sum() / control_mask.sum()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, device, use_amp=True):
    """Validate the model with mixed precision."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_sequence = batch['input_sequence'].to(device, non_blocking=True)
            controls_target = batch['control'].to(device, non_blocking=True)
            control_mask = batch['control_mask'].to(device, non_blocking=True)
            
            # Use mixed precision for validation too
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    controls_pred = model(input_sequence, control_mask=control_mask)
                    controls_pred_last = controls_pred[:, -1, :]
                    
                    loss_per_dim = (controls_pred_last - controls_target) ** 2
                    masked_loss = loss_per_dim * control_mask
                    loss = masked_loss.sum() / control_mask.sum()
            else:
                controls_pred = model(input_sequence, control_mask=control_mask)
                controls_pred_last = controls_pred[:, -1, :]
                
                loss_per_dim = (controls_pred_last - controls_target) ** 2
                masked_loss = loss_per_dim * control_mask
                loss = masked_loss.sum() / control_mask.sum()
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    # Set seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    py_random.seed(RANDOM_SEED)
    
    # Device - use GPU(s) if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        n_gpus = torch.cuda.device_count()
        print(f"Using device: cuda with {n_gpus} GPU(s)")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU
        n_gpus = 1
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        n_gpus = 0
        print(f"Using device: {device}")
    
    # Load data with memory-efficient dataset
    data_file = os.path.join(DATA_DIR, 'lqr_training_data.pkl')
    
    print("\n=== Creating Memory-Efficient Datasets ===")
    # Use ALL data for proper training
    dataset = MemoryEfficientLQRDataset(
        data_file,
        sequence_length=SEQUENCE_LENGTH,
        normalization=NORMALIZATION_STRATEGY,
        subset_ratio=1.0  # Use all data
    )
    
    # Split into train (95%) / test (5%)
    test_size = int(len(dataset) * 0.05)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    print(f"Train size: {train_size:,} (95%)")
    print(f"Test size: {test_size:,} (5%)")
    
    # Create dataloaders
    # Use many workers for fast data loading on multi-GPU server
    num_workers = 16 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    prefetch_factor = 4 if torch.cuda.is_available() else 2
    
    print(f"DataLoader config: num_workers={num_workers}, pin_memory={pin_memory}, prefetch_factor={prefetch_factor}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Create model
    print("\n=== Creating Model ===")
    model = UniversalLQRTransformer(
        max_state_dim=MAX_STATE_DIM,
        max_control_dim=MAX_INPUT_DIM,
        dimension_encoding_size=DIMENSION_ENCODING_SIZE,
        d_model=TRANSFORMER_CONFIG['d_model'],
        n_heads=TRANSFORMER_CONFIG['n_heads'],
        n_layers=TRANSFORMER_CONFIG['n_layers'],
        d_ff=TRANSFORMER_CONFIG['d_ff'],
        dropout=TRANSFORMER_CONFIG['dropout'],
        max_seq_len=TRANSFORMER_CONFIG['max_seq_len']
    )
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Enable multi-GPU training if available
    if n_gpus > 1:
        print(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    
    # Training loop
    print("\n=== Starting Training ===")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
        
        # Train with mixed precision
        train_loss = train_epoch(model, train_loader, optimizer, device, use_amp=True)
        history['train_loss'].append(train_loss)
        
        # Test with mixed precision
        test_loss = validate(model, test_loader, device, use_amp=True)
        history['val_loss'].append(test_loss)
        
        print(f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        
        # Save best model
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            # Save model (handle DataParallel wrapper)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))
            print(f"[OK] Saved best model (test_loss: {test_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % TRAINING_CONFIG['save_every'] == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch+1}.pt'))
    
    print("\n=== Training Complete ===")
    print(f"Best test loss: {best_val_loss:.6f}")
    print(f"Model saved to: {MODEL_DIR}")
    
    # Save training history
    import json
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
