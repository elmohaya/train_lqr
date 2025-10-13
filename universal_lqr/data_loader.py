"""
Efficient Data Loader for JAX
Supports streaming from HDF5 files for memory-efficient training

Design Philosophy:
- Data loader uses NumPy (HDF5 files are NumPy-based)
- Conversion to JAX happens via jax.device_put() before training
- Separation of I/O (NumPy) and computation (JAX) concerns
"""

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple

from config import DATA_DIR, TRAINING_CONFIG


class JAXDataLoader:
    """
    JAX-compatible data loader that streams from HDF5.
    
    Returns NumPy arrays by default (for I/O efficiency).
    Use to_jax=True or manually convert with jax.device_put() for GPU transfer.
    
    Example:
        loader = JAXDataLoader(path, to_jax=True)  # Auto-convert to JAX
        # OR
        loader = JAXDataLoader(path, to_jax=False)
        batch = next(iter(loader))
        batch_jax = jax.tree.map(jax.device_put, batch)  # Manual conversion
    """
    
    def __init__(self, h5_path: str, split: str = 'train', 
                 batch_size: int = 1024, validation_split: float = 0.15,
                 shuffle: bool = True, to_jax: bool = False, device: str = None):
        """
        Args:
            h5_path: Path to HDF5 file
            split: 'train' or 'val'
            batch_size: Batch size
            validation_split: Fraction for validation
            shuffle: Whether to shuffle data
            to_jax: If True, automatically convert batches to JAX arrays
            device: Device to put JAX arrays on (e.g., 'gpu', 'cpu'). None = default
        """
        self.h5_path = h5_path
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_jax = to_jax
        self.device = device
        
        # Open file and get metadata
        with h5py.File(h5_path, 'r') as f:
            self.total_sequences = f.attrs['total_sequences']
            self.seq_length = f.attrs['sequence_length']
            self.max_state_dim = f.attrs['max_state_dim']
            self.max_input_dim = f.attrs['max_input_dim']
            
            # Load normalization statistics
            self.state_mean = f['state_mean'][:]
            self.state_std = f['state_std'][:]
            self.control_mean = f['control_mean'][:]
            self.control_std = f['control_std'][:]
        
        # Split indices
        n_val = int(self.total_sequences * validation_split)
        n_train = self.total_sequences - n_val
        
        np.random.seed(42)
        indices = np.random.permutation(self.total_sequences)
        
        if split == 'train':
            self.indices = indices[:n_train]
        else:
            self.indices = indices[n_train:]
        
        self.n_batches = len(self.indices) // batch_size
        
        # Open file handle
        self.h5_file = h5py.File(h5_path, 'r')
        self.input_sequences = self.h5_file['input_sequences']
        self.controls = self.h5_file['controls']
        self.control_masks = self.h5_file['control_masks']
        
        # Shuffle indices
        if shuffle:
            np.random.shuffle(self.indices)
        
        self.current_batch = 0
    
    def __len__(self):
        return self.n_batches
    
    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.current_batch >= self.n_batches:
            raise StopIteration
        
        # Get batch indices
        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load batch (use advanced indexing)
        batch_indices_sorted = np.sort(batch_indices)  # Sort for faster HDF5 access
        
        input_seq = self.input_sequences[batch_indices_sorted]
        control = self.controls[batch_indices_sorted]
        control_mask = self.control_masks[batch_indices_sorted]
        
        # Unsort to match original order if shuffled
        if self.shuffle:
            unsort_indices = np.argsort(np.argsort(batch_indices))
            input_seq = input_seq[unsort_indices]
            control = control[unsort_indices]
            control_mask = control_mask[unsort_indices]
        
        # ⚠️ NOTE: Data is ALREADY normalized by per-system max values during generation
        # The stored mean/std are dummy values (mean=0, std=1) for backward compatibility
        # No additional normalization needed here!
        # 
        # Each system was normalized by its own max values:
        #   x_normalized = x_raw / x_max_for_system
        #   u_normalized = u_raw / u_max_for_system
        #
        # To denormalize transformer outputs, use the normalization factors
        # saved in the HDF5 file under 'normalization/{system_name}/control_max'
        #
        # If you want to verify, the current mean/std will do nothing:
        # input_seq_states_norm = (input_seq_states - 0.0) / 1.0 = input_seq_states
        # control_norm = (control - 0.0) / 1.0 = control
        
        self.current_batch += 1
        
        # Create batch dict (numpy arrays, now normalized)
        batch = {
            'input_sequences': input_seq,
            'controls': control,
            'control_masks': control_mask
        }
        
        # Convert to JAX if requested
        if self.to_jax:
            if self.device:
                # Put on specific device
                device_obj = jax.devices(self.device)[0]
                batch = jax.tree.map(lambda x: jax.device_put(x, device_obj), batch)
            else:
                # Put on default device (GPU if available)
                batch = jax.tree.map(jax.device_put, batch)
        
        return batch
    
    def get_normalization_stats(self):
        """Return normalization statistics."""
        return {
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'control_mean': self.control_mean,
            'control_std': self.control_std
        }
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def create_jax_dataloaders(h5_path: str,
                          batch_size: int,
                          validation_split: float = 0.15,
                          to_jax: bool = True,
                          device: str = None) -> Tuple[JAXDataLoader, JAXDataLoader]:
    """
    Create JAX data loaders for training and validation.
    
    Args:
        h5_path: Path to HDF5 file
        batch_size: Batch size
        validation_split: Fraction for validation
        to_jax: If True, automatically convert to JAX arrays (recommended for training)
        device: Device to put arrays on ('gpu', 'cpu', or None for default)
    
    Returns:
        train_loader, val_loader
        
    Note:
        - If to_jax=True: Returns JAX arrays on device (ready for training)
        - If to_jax=False: Returns NumPy arrays (convert manually with jax.device_put)
    """
    train_loader = JAXDataLoader(
        h5_path, 
        split='train', 
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        to_jax=to_jax,
        device=device
    )
    
    val_loader = JAXDataLoader(
        h5_path,
        split='val',
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,
        to_jax=to_jax,
        device=device
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    """Test data loaders"""
    import os
    
    h5_path = os.path.join(DATA_DIR, 'training_data.h5')
    
    if not os.path.exists(h5_path):
        print(f"Data file not found: {h5_path}")
        print("Please run data_generation_jax.py first.")
        exit(1)
    
    print("="*70)
    print("Testing JAX Data Loader")
    print("="*70)
    
    train_loader, val_loader = create_jax_dataloaders(
        h5_path,
        batch_size=TRAINING_CONFIG['batch_size'],
        validation_split=TRAINING_CONFIG['validation_split']
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Load one batch
    batch = next(iter(train_loader))
    print(f"\nBatch input shape: {batch['input_sequences'].shape}")
    print(f"Batch control shape: {batch['controls'].shape}")
    print(f"Batch mask shape: {batch['control_masks'].shape}")
    
    # Test normalization stats
    stats = train_loader.get_normalization_stats()
    print(f"\nNormalization Statistics:")
    print(f"  State mean shape: {stats['state_mean'].shape}")
    print(f"  State std shape: {stats['state_std'].shape}")
    print(f"  Control mean shape: {stats['control_mean'].shape}")
    print(f"  Control std shape: {stats['control_std'].shape}")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)

