"""
Inspect Generated Data
Reads and displays random samples from the HDF5 data file in input-output format.
"""

import h5py
import numpy as np
import sys
import os

def inspect_data(h5_path, num_samples=200, detailed_samples=5):
    """
    Read and display random samples from the generated data.
    
    Args:
        h5_path: Path to HDF5 file
        num_samples: Number of samples to read
        detailed_samples: Number of samples to show in full detail
    """
    
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return
    
    print("="*80)
    print("DATA INSPECTION REPORT")
    print("="*80)
    print(f"File: {h5_path}")
    print(f"File size: {os.path.getsize(h5_path) / 1e6:.2f} MB")
    print()
    
    with h5py.File(h5_path, 'r') as f:
        # Print metadata
        print("="*80)
        print("METADATA")
        print("="*80)
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print()
        
        # Print dataset info
        print("="*80)
        print("DATASETS")
        print("="*80)
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"  {key}: {f[key].shape} {f[key].dtype}")
        print()
        
        # Get dataset info (without loading data)
        total_samples = f['input_sequences'].shape[0]
        print(f"Total samples in dataset: {total_samples:,}")
        print()
        
        # Sample random indices
        num_samples = min(num_samples, total_samples)
        random_indices = np.random.choice(total_samples, size=num_samples, replace=False)
        random_indices = np.sort(random_indices)  # Sort for easier reading
        
        # Read ONLY the sampled data (not the entire file!)
        print(f"\n{'='*80}")
        print(f"EFFICIENT SAMPLING")
        print(f"{'='*80}")
        print(f"Reading only {num_samples} samples from file...")
        print(f"(NOT loading entire dataset - memory efficient!)")
        
        input_sequences = f['input_sequences'][random_indices]
        controls = f['controls'][random_indices]
        control_masks = f['control_masks'][random_indices]
        
        # Estimate memory usage
        memory_mb = (input_sequences.nbytes + controls.nbytes + control_masks.nbytes) / 1e6
        total_memory_mb = (f['input_sequences'].shape[0] * f['input_sequences'][0].nbytes + 
                          f['controls'].shape[0] * f['controls'][0].nbytes +
                          f['control_masks'].shape[0] * f['control_masks'][0].nbytes) / 1e6
        print(f"Memory loaded: {memory_mb:.2f} MB (would be {total_memory_mb:.2f} MB for full dataset)")
        print(f"Memory savings: {100 * (1 - memory_mb/total_memory_mb):.1f}%")
        print()
        
        # Get normalization stats if available
        if 'state_mean' in f:
            state_mean = f['state_mean'][:]
            state_std = f['state_std'][:]
            control_mean = f['control_mean'][:]
            control_std = f['control_std'][:]
            has_stats = True
        else:
            has_stats = False
        
        # Statistics
        input_norms = []
        control_norms = []
        active_dims = []
        
        print("="*80)
        print(f"ANALYZING {num_samples} RANDOM SEQUENCES")
        print("="*80)
        
        for i in range(len(input_sequences)):
            input_seq = input_sequences[i]
            control = controls[i]
            mask = control_masks[i]
            original_idx = random_indices[i]
            
            # Get active dimensions
            n_u = int(np.sum(mask))
            active_dims.append(n_u)
            
            # Get active control (unpadded)
            active_control = control[:n_u]
            
            # Compute norms
            input_norms.append(np.linalg.norm(input_seq))
            control_norms.append(np.linalg.norm(active_control))
            
            # Show detailed info for first few samples
            if i < detailed_samples:
                print(f"\n{'─'*80}")
                print(f"SAMPLE {i+1} (Original Index: {original_idx})")
                print(f"{'─'*80}")
                
                print(f"\nControl Dimension: {n_u}")
                print(f"Control Mask: {mask.astype(int)}")
                
                print(f"\nINPUT SEQUENCE (shape: {input_seq.shape}):")
                print(f"  First timestep:  {input_seq[0]}")
                print(f"  Last timestep:   {input_seq[-1]}")
                print(f"  Sequence norm:   {np.linalg.norm(input_seq):.4f}")
                print(f"  Min value:       {np.min(input_seq):.4f}")
                print(f"  Max value:       {np.max(input_seq):.4f}")
                print(f"  Mean value:      {np.mean(input_seq):.4f}")
                print(f"  Std value:       {np.std(input_seq):.4f}")
                
                print(f"\nOUTPUT CONTROL (shape: {control.shape}):")
                print(f"  Padded control:   {control}")
                print(f"  Active control:   {active_control}")
                print(f"  Control norm:     {np.linalg.norm(active_control):.4f}")
                print(f"  Control mask:     {mask}")
                
                # Show full input sequence structure
                print(f"\nFULL INPUT SEQUENCE (all timesteps):")
                seq_len = input_seq.shape[0]
                for t in range(min(3, seq_len)):  # Show first 3 timesteps
                    print(f"  t={t}: {input_seq[t]}")
                if seq_len > 6:
                    print(f"  ... ({seq_len - 6} timesteps omitted) ...")
                for t in range(max(3, seq_len - 3), seq_len):  # Show last 3 timesteps
                    if t >= 3:
                        print(f"  t={t}: {input_seq[t]}")
        
        # Summary statistics
        print(f"\n{'='*80}")
        print(f"SUMMARY STATISTICS (from {num_samples} samples)")
        print(f"{'='*80}")
        
        print(f"\nInput Sequence Norms:")
        print(f"  Mean:   {np.mean(input_norms):.4f}")
        print(f"  Std:    {np.std(input_norms):.4f}")
        print(f"  Min:    {np.min(input_norms):.4f}")
        print(f"  Max:    {np.max(input_norms):.4f}")
        
        print(f"\nControl Norms:")
        print(f"  Mean:   {np.mean(control_norms):.4f}")
        print(f"  Std:    {np.std(control_norms):.4f}")
        print(f"  Min:    {np.min(control_norms):.4f}")
        print(f"  Max:    {np.max(control_norms):.4f}")
        
        print(f"\nActive Control Dimensions:")
        unique_dims = np.unique(active_dims)
        for dim in unique_dims:
            count = np.sum(np.array(active_dims) == dim)
            percentage = (count / len(active_dims)) * 100
            print(f"  {dim}D control: {count} samples ({percentage:.1f}%)")
        
        # Check for NaN or Inf (in sampled data)
        has_nan_input = np.any(np.isnan(input_sequences))
        has_inf_input = np.any(np.isinf(input_sequences))
        has_nan_control = np.any(np.isnan(controls))
        has_inf_control = np.any(np.isinf(controls))
        
        print(f"\nData Quality:")
        print(f"  Input contains NaN:  {has_nan_input}")
        print(f"  Input contains Inf:  {has_inf_input}")
        print(f"  Control contains NaN: {has_nan_control}")
        print(f"  Control contains Inf: {has_inf_control}")
        
        # Normalization stats
        if has_stats:
            print(f"\n{'='*80}")
            print(f"NORMALIZATION STATISTICS")
            print(f"{'='*80}")
            print(f"\nState Mean (first 10 dims): {state_mean[:10]}")
            print(f"State Std (first 10 dims):  {state_std[:10]}")
            print(f"\nControl Mean: {control_mean}")
            print(f"Control Std:  {control_std}")
        
        print(f"\n{'='*80}")
        print(f"DATA FORMAT EXPLANATION")
        print(f"{'='*80}")
        print(f"""
The data is organized as:

INPUT SEQUENCE: [sequence_length, state_dim + control_dim + system_encoding]
  - Contains sequence of past states with dimension encodings
  - Shape: [{f.attrs.get('sequence_length', 'N/A')}, {input_seq.shape[1]}]
  
OUTPUT CONTROL: [max_control_dim]
  - The control action to predict (padded to max dimension)
  - Shape: [{control.shape[0]}]
  
CONTROL MASK: [max_control_dim]
  - Binary mask indicating active control dimensions
  - 1 = active dimension, 0 = padding
  - Shape: [{mask.shape[0]}]

Training:
  - Input: state sequence → Output: next control action
  - Loss is computed only on active dimensions using the mask
  - This allows handling systems with different control dimensions
        """)
        
        print(f"\n{'='*80}")
        print(f"✓ DATA INSPECTION COMPLETE")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    # Default path
    default_path = os.path.join(
        os.path.dirname(__file__), 
        'data', 
        'test_jax_quick.h5'
    )
    
    # Check if user provided path
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    else:
        h5_path = default_path
    
    # Check if user provided number of samples
    num_samples = 200
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])
    
    # Check if user provided number of detailed samples
    detailed_samples = 5
    if len(sys.argv) > 3:
        detailed_samples = int(sys.argv[3])
    
    print(f"""
{'='*80}
USAGE:
  python inspect_data.py [h5_file] [num_samples] [detailed_samples]

ARGUMENTS:
  h5_file:         Path to HDF5 data file (default: data/test_jax_quick.h5)
  num_samples:     Number of random samples to analyze (default: 200)
  detailed_samples: Number of samples to show in detail (default: 5)

EXAMPLES:
  python inspect_data.py
  python inspect_data.py data/training_data.h5
  python inspect_data.py data/training_data.h5 500 10
{'='*80}
    """)
    
    inspect_data(h5_path, num_samples, detailed_samples)

