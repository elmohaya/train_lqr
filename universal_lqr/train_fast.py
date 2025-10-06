"""
FAST Training Script using Pre-Processed Data

Run preprocess_data.py ONCE first, then use this script.
This is 50-100x faster than train_memory_efficient.py!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
from tqdm import tqdm
import random as py_random

from config import (
    TRANSFORMER_CONFIG, TRAINING_CONFIG,
    DATA_DIR, MODEL_DIR, RANDOM_SEED,
    MAX_STATE_DIM, MAX_INPUT_DIM, DIMENSION_ENCODING_SIZE
)
from transformer_model import UniversalLQRTransformer, count_parameters


class FastH5Dataset(Dataset):
    """Lightning-fast dataset that reads pre-processed HDF5 data."""
    
    def __init__(self, h5_file, indices):
        """
        Args:
            h5_file: Path to preprocessed HDF5 file
            indices: List of indices to use (for train/test split)
        """
        self.h5_file = h5_file
        self.indices = indices
        self.h5f = None
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Lazy open HDF5 file (for multi-processing)
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_file, 'r')
        
        real_idx = self.indices[idx]
        
        return {
            'input_sequence': torch.from_numpy(self.h5f['input_sequences'][real_idx]),
            'control': torch.from_numpy(self.h5f['controls'][real_idx]),
            'control_mask': torch.from_numpy(self.h5f['control_masks'][real_idx])
        }


def train_epoch(model, dataloader, optimizer, device, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    for batch in tqdm(dataloader, desc="Training", dynamic_ncols=True):
        input_sequence = batch['input_sequence'].to(device, non_blocking=True)
        controls_target = batch['control'].to(device, non_blocking=True)
        control_mask = batch['control_mask'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                controls_pred = model(input_sequence, control_mask=control_mask)
                controls_pred_last = controls_pred[:, -1, :]
                
                loss_per_dim = (controls_pred_last - controls_target) ** 2
                masked_loss = loss_per_dim * control_mask
                loss = masked_loss.sum() / control_mask.sum()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            controls_pred = model(input_sequence, control_mask=control_mask)
            controls_pred_last = controls_pred[:, -1, :]
            
            loss_per_dim = (controls_pred_last - controls_target) ** 2
            masked_loss = loss_per_dim * control_mask
            loss = masked_loss.sum() / control_mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, device, use_amp=True):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", dynamic_ncols=True):
            input_sequence = batch['input_sequence'].to(device, non_blocking=True)
            controls_target = batch['control'].to(device, non_blocking=True)
            control_mask = batch['control_mask'].to(device, non_blocking=True)
            
            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
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
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        n_gpus = torch.cuda.device_count()
        print(f"Using device: cuda with {n_gpus} GPU(s)")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device('cpu')
        n_gpus = 0
        print(f"Using device: cpu")
    
    # Load preprocessed data
    h5_file = os.path.join(DATA_DIR, 'preprocessed_sequences.h5')
    
    if not os.path.exists(h5_file):
        print(f"\nERROR: Preprocessed file not found: {h5_file}")
        print("Please run: python preprocess_data.py")
        print("This will take ~10-20 minutes but only needs to run ONCE.")
        return
    
    print(f"\nLoading preprocessed data from: {h5_file}")
    with h5py.File(h5_file, 'r') as hf:
        total_sequences = hf['input_sequences'].shape[0]
        print(f"Total sequences: {total_sequences:,}")
    
    # Create train/test split
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    
    test_size = int(total_sequences * 0.05)
    train_size = total_sequences - test_size
    
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]
    
    print(f"Train size: {train_size:,} (95%)")
    print(f"Test size: {test_size:,} (5%)")
    
    # Create datasets
    train_dataset = FastH5Dataset(h5_file, train_indices)
    test_dataset = FastH5Dataset(h5_file, test_indices)
    
    # Create dataloaders
    num_workers = 8 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    
    print(f"\nDataLoader config: batch_size={TRAINING_CONFIG['batch_size']}, num_workers={num_workers}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Batches per epoch: {len(train_loader):,}")
    
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
    
    # Enable multi-GPU
    if n_gpus > 1:
        print(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    
    # Training loop
    print("\n=== Starting Training ===\n")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, use_amp=True)
        history['train_loss'].append(train_loss)
        
        # Test
        test_loss = validate(model, test_loader, device, use_amp=True)
        history['val_loss'].append(test_loss)
        
        print(f"\nTrain Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        
        # Save best model
        if test_loss < best_val_loss:
            best_val_loss = test_loss
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
            print(f"[OK] Saved checkpoint at epoch {epoch+1}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Best test loss: {best_val_loss:.6f}")
    print(f"Model saved to: {MODEL_DIR}")
    print("="*70)
    
    # Save training history
    import json
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()

