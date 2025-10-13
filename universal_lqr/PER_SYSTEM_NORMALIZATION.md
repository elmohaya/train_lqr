# Per-System Normalization Implementation

## Problem Identified

**The user correctly identified** that normalizing all systems together by global mean/std was fundamentally wrong because:
- Different LTI systems have different physical units (meters, radians, Newtons, etc.)
- A pendulum angle (radians) shouldn't be normalized the same way as a vehicle position (meters)
- Control signals have vastly different magnitudes across systems

## Solution Implemented

### Per-System Normalization by Actual Maximum Values

1. **During Data Generation**:
   - Track actual max |x| and max |u| encountered during simulation for EACH system
   - Store raw (unnormalized) data initially
   - After all data is generated, normalize by system-specific maximums
   - Save normalization factors for later denormalization

2. **Normalization Formula**:
   ```
   x_normalized = x_raw / x_max_for_this_system
   u_normalized = u_raw / u_max_for_this_system
   ```

3. **Data Stays in [-1, 1] Range**:
   - All systems are now on comparable scales
   - Transformer learns on normalized data
   - Can denormalize outputs using saved max values

## Changes Made to `data_generation_jax.py`

### 1. Track System ID
```python
# Added new HDF5 dataset
ds_system_id = f.create_dataset(
    'system_ids',
    shape=(0,),
    maxshape=(estimated_sequences,),
    dtype='S50',  # String up to 50 chars
)
```

### 2. Track Max Values Per System
```python
# Initialize for each system
system_max_values[system_name] = {
    'state_max': np.zeros(n_x),
    'control_max': np.zeros(n_u),
    'n_states': n_x,
    'n_inputs': n_u
}

# Update during simulation
system_max_values[system_name]['state_max'] = np.maximum(
    system_max_values[system_name]['state_max'],
    np.max(np.abs(X), axis=0)
)
```

### 3. Normalize After Generation
```python
# Process all sequences in chunks
for each sequence:
    sys_name = system_ids[i]
    state_max = system_max_values[sys_name]['state_max']
    control_max = system_max_values[sys_name]['control_max']
    
    # Normalize
    input_seqs[i, :, :n_x] /= state_max
    controls[i, :n_u] /= control_max
```

### 4. Save Normalization Factors
```python
# Create HDF5 group structure
norm_group = f.create_group('normalization')
for system_name in system_max_values:
    sys_group = norm_group.create_group(system_name)
    sys_group.create_dataset('state_max', data=...)
    sys_group.create_dataset('control_max', data=...)
```

## HDF5 File Structure

```
training_data_jax.h5
├── input_sequences         # (N, seq_len, state_dim) - NORMALIZED
├── controls                # (N, control_dim) - NORMALIZED
├── control_masks           # (N, control_dim)
├── system_ids              # (N,) - system name for each sequence
├── normalization/          # Per-system max values
│   ├── InvertedPendulum/
│   │   ├── state_max       # Max state values for this system
│   │   ├── control_max     # Max control values for this system
│   │   └── attrs: n_states, n_inputs
│   ├── CartPole/
│   │   ├── state_max
│   │   └── control_max
│   └── ...
└── state_mean, state_std, control_mean, control_std  # Dummy (for compatibility)
```

## How to Denormalize Transformer Outputs

When the transformer predicts control for a specific system:

```python
import h5py

# Load normalization factors
with h5py.File('training_data_jax.h5', 'r') as f:
    system_name = 'InvertedPendulum'  # or whatever system
    control_max = f[f'normalization/{system_name}/control_max'][:]

# Transformer outputs normalized control
u_normalized = transformer_output  # Range: ~[-1, 1]

# Denormalize to get actual control
u_actual = u_normalized * control_max
```

## Benefits

1. **Proper Physical Units**: Each system normalized by its own scale
2. **Comparable Training**: All systems contribute equally to transformer learning
3. **No Mixing Units**: Pendulum angles don't pollute vehicle positions
4. **Easy Denormalization**: Save max values for inference
5. **Better Generalization**: Transformer learns system-agnostic control patterns

## Additional Changes

- **Increased Parameter Uncertainty**: ±10% → ±20%
  ```python
  PARAMETER_UNCERTAINTY = 0.20  # Override: ±20% parameter variation
  ```

## Data Loader Impact

The data loader (`data_loader.py`) should be updated to:
- **NOT normalize** data (already normalized)
- Or use dummy normalization (divide by 1.0, subtract 0.0)
- The stored `state_mean/std` and `control_mean/std` are now dummy values

## Testing

To verify normalization:
```python
import h5py
import numpy as np

with h5py.File('training_data_jax.h5', 'r') as f:
    # Check that data is in reasonable range
    inputs = f['input_sequences'][:1000]
    controls = f['controls'][:1000]
    
    print(f"Input range: [{np.min(inputs):.2f}, {np.max(inputs):.2f}]")
    print(f"Control range: [{np.min(controls):.2f}, {np.max(controls):.2f}]")
    # Should be roughly [-1, 1] with some outliers
    
    # Check normalization factors exist
    system_names = list(f['normalization'].keys())
    print(f"\nNormalization factors saved for {len(system_names)} systems")
    for sys_name in system_names[:5]:
        state_max = f[f'normalization/{sys_name}/state_max'][:]
        control_max = f[f'normalization/{sys_name}/control_max'][:]
        print(f"{sys_name}: state_max={np.max(state_max):.2e}, control_max={np.max(control_max):.2e}")
```

---

**Status**: ✅ Implemented and ready for data generation!

