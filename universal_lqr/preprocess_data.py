"""
Pre-process all sequences and save to efficient HDF5 format.
Run this ONCE, then training loads pre-processed data instantly.
"""

import numpy as np
import h5py
import pickle
import os
from tqdm import tqdm

from config import (
    SEQUENCE_LENGTH, DATA_DIR, MAX_STATE_DIM, MAX_INPUT_DIM,
    DIMENSION_ENCODING_SIZE, NORMALIZATION_STRATEGY
)
from data_utils import create_dimension_encoding


def preprocess_all_sequences(data_file, output_file, sequence_length=32):
    """
    Pre-process ALL sequences and save to HDF5.
    This takes ~10-20 minutes but only needs to run ONCE.
    """
    print("="*70)
    print("PRE-PROCESSING ALL SEQUENCES")
    print("="*70)
    print(f"Loading data from: {data_file}")
    
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"Loaded {len(all_data)} system variants")
    
    # First pass: count total sequences
    print("\nCounting sequences...")
    total_sequences = 0
    for system_data in tqdm(all_data):
        for trajectory in system_data['trajectories']:
            traj_len = len(trajectory['time'])
            n_sequences = traj_len - sequence_length
            if n_sequences > 0:
                total_sequences += n_sequences
    
    print(f"Total sequences: {total_sequences:,}")
    print(f"Estimated size: {total_sequences * sequence_length * (MAX_STATE_DIM + DIMENSION_ENCODING_SIZE) * 4 / (1024**3):.2f} GB")
    
    # Compute normalization statistics
    print("\nComputing normalization statistics...")
    n_samples = min(10000, total_sequences)
    sample_indices = np.random.choice(total_sequences, n_samples, replace=False)
    sample_set = set(sample_indices)
    
    state_values = [[] for _ in range(MAX_STATE_DIM)]
    control_values = [[] for _ in range(MAX_INPUT_DIM)]
    
    seq_idx = 0
    for system_data in all_data:
        n_states = system_data['n_states']
        n_inputs = system_data['n_inputs']
        
        for trajectory in system_data['trajectories']:
            traj_len = len(trajectory['time'])
            n_sequences = traj_len - sequence_length
            
            for seq_start in range(n_sequences):
                if seq_idx in sample_set:
                    states = trajectory['states'][seq_start:seq_start + sequence_length]
                    control = trajectory['controls'][seq_start + sequence_length]
                    
                    for dim in range(n_states):
                        state_values[dim].extend(states[:, dim])
                    for dim in range(n_inputs):
                        control_values[dim].append(control[dim])
                
                seq_idx += 1
    
    # Compute mean and std
    state_mean = np.zeros(MAX_STATE_DIM)
    state_std = np.ones(MAX_STATE_DIM)
    control_mean = np.zeros(MAX_INPUT_DIM)
    control_std = np.ones(MAX_INPUT_DIM)
    
    for dim in range(MAX_STATE_DIM):
        if len(state_values[dim]) > 0:
            state_mean[dim] = np.mean(state_values[dim])
            state_std[dim] = np.std(state_values[dim]) + 1e-8
    
    for dim in range(MAX_INPUT_DIM):
        if len(control_values[dim]) > 0:
            control_mean[dim] = np.mean(control_values[dim])
            control_std[dim] = np.std(control_values[dim]) + 1e-8
    
    print("Statistics computed.")
    
    # Create HDF5 file
    print(f"\nCreating HDF5 file: {output_file}")
    print("This will take 10-20 minutes...")
    
    with h5py.File(output_file, 'w') as hf:
        # Create datasets
        input_sequences = hf.create_dataset(
            'input_sequences',
            shape=(total_sequences, sequence_length, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE),
            dtype='float32',
            chunks=(1024, sequence_length, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE),
            compression='lzf'
        )
        
        controls = hf.create_dataset(
            'controls',
            shape=(total_sequences, MAX_INPUT_DIM),
            dtype='float32',
            chunks=(1024, MAX_INPUT_DIM),
            compression='lzf'
        )
        
        control_masks = hf.create_dataset(
            'control_masks',
            shape=(total_sequences, MAX_INPUT_DIM),
            dtype='float32',
            chunks=(1024, MAX_INPUT_DIM),
            compression='lzf'
        )
        
        # Store normalization stats
        hf.create_dataset('state_mean', data=state_mean)
        hf.create_dataset('state_std', data=state_std)
        hf.create_dataset('control_mean', data=control_mean)
        hf.create_dataset('control_std', data=control_std)
        
        # Process and save all sequences (with batching for speed)
        seq_idx = 0
        batch_size = 10000
        
        # Temporary buffers for batching
        input_batch = []
        control_batch = []
        mask_batch = []
        
        for system_idx, system_data in enumerate(tqdm(all_data, desc="Processing systems")):
            n_states = system_data['n_states']
            n_inputs = system_data['n_inputs']
            
            # Create dimension encoding once per system
            dim_encoding = create_dimension_encoding(n_inputs, n_states)
            if hasattr(dim_encoding, 'numpy'):
                dim_encoding = dim_encoding.numpy()
            dim_encoding_repeated = np.tile(dim_encoding, (sequence_length, 1))
            
            # Control mask for this system
            control_mask = np.zeros(MAX_INPUT_DIM, dtype=np.float32)
            control_mask[:n_inputs] = 1.0
            
            for traj_idx, trajectory in enumerate(system_data['trajectories']):
                traj_len = len(trajectory['time'])
                n_sequences = traj_len - sequence_length
                
                if n_sequences <= 0:
                    continue
                
                # Process all sequences from this trajectory efficiently
                for seq_start in range(n_sequences):
                    # Extract and normalize states
                    states = trajectory['states'][seq_start:seq_start + sequence_length].copy()
                    states[:, :n_states] = (states[:, :n_states] - state_mean[:n_states]) / state_std[:n_states]
                    
                    # Pad states
                    states_padded = np.zeros((sequence_length, MAX_STATE_DIM), dtype=np.float32)
                    states_padded[:, :n_states] = states
                    
                    # Concatenate with dimension encoding
                    input_seq = np.concatenate([states_padded, dim_encoding_repeated], axis=1)
                    
                    # Extract and normalize control
                    control = trajectory['controls'][seq_start + sequence_length].copy()
                    control[:n_inputs] = (control[:n_inputs] - control_mean[:n_inputs]) / control_std[:n_inputs]
                    
                    # Pad control
                    control_padded = np.zeros(MAX_INPUT_DIM, dtype=np.float32)
                    control_padded[:n_inputs] = control
                    
                    # Add to batch
                    input_batch.append(input_seq)
                    control_batch.append(control_padded)
                    mask_batch.append(control_mask)
                    
                    # Write batch when full
                    if len(input_batch) >= batch_size:
                        batch_start = seq_idx
                        batch_end = seq_idx + len(input_batch)
                        
                        input_sequences[batch_start:batch_end] = np.array(input_batch)
                        controls[batch_start:batch_end] = np.array(control_batch)
                        control_masks[batch_start:batch_end] = np.array(mask_batch)
                        
                        seq_idx = batch_end
                        input_batch = []
                        control_batch = []
                        mask_batch = []
            
            # Print progress every 50 systems
            if (system_idx + 1) % 50 == 0:
                print(f"  Progress: {seq_idx:,} sequences processed...")
        
        # Write remaining sequences
        if len(input_batch) > 0:
            batch_start = seq_idx
            batch_end = seq_idx + len(input_batch)
            
            input_sequences[batch_start:batch_end] = np.array(input_batch)
            controls[batch_start:batch_end] = np.array(control_batch)
            control_masks[batch_start:batch_end] = np.array(mask_batch)
            
            seq_idx = batch_end
        
        print(f"\nProcessed {seq_idx:,} sequences")
    
    print(f"\n{'='*70}")
    print(f"PRE-PROCESSING COMPLETE!")
    print(f"Saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")
    print(f"{'='*70}")


if __name__ == '__main__':
    data_file = os.path.join(DATA_DIR, 'lqr_training_data.pkl')
    output_file = os.path.join(DATA_DIR, 'preprocessed_sequences.h5')
    
    preprocess_all_sequences(data_file, output_file, SEQUENCE_LENGTH)

