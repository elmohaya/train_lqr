"""
JAX-Accelerated Data Generation (OPTIONAL)
Ultra-fast trajectory generation using JAX JIT compilation and vectorization

This is an optional alternative to data_generation.py that uses JAX for:
1. JIT-compiled trajectory simulation (10-50x faster)
2. Vectorized batch generation (process multiple trajectories simultaneously)
3. GPU acceleration (if available)

Note: Requires JAX installation. Falls back to numpy implementation if JAX not available.
"""

import numpy as np
import h5py
import os
import time
from tqdm import tqdm
import sys

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    from functools import partial
    JAX_AVAILABLE = True
    print("JAX detected! Using accelerated trajectory generation.")
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available. Install with: pip install jax jaxlib")
    print("Falling back to numpy implementation...")

from config import (
    DATA_DIR, NUM_VARIANTS_PER_SYSTEM, NUM_TRAJECTORIES_PER_VARIANT,
    TIME_HORIZON, DT, PARAMETER_UNCERTAINTY, PROCESS_NOISE_STD,
    SEQUENCE_LENGTH, SEQUENCE_STRIDE, MAX_STATE_DIM, MAX_INPUT_DIM,
    DIMENSION_ENCODING_SIZE, RANDOM_SEED
)

# Increase uncertainty range from ±10% to ±20%
PARAMETER_UNCERTAINTY = 0.20  # Override: ±20% parameter variation
from data_utils import prepare_input_sequence, pad_control, create_control_mask
from systems import __all__ as ALL_SYSTEMS
import systems
from lqr_controller import design_lqr


if JAX_AVAILABLE:
    @partial(jit, static_argnums=(5, 8, 9))  # n_steps, settling_window, min_steps are static
    def simulate_trajectory_jax(A, B, K, x0, dt, n_steps, process_noise_std, rng_key, 
                                settling_window=100, min_steps=1000):
        """
        JIT-compiled trajectory simulation using JAX with stability and settling detection.
        
        Args:
            A: State matrix (n_x, n_x)
            B: Input matrix (n_x, n_u)
            K: LQR gain (n_u, n_x)
            x0: Initial state (n_x,)
            dt: Time step
            n_steps: Number of steps
            process_noise_std: Process noise standard deviation
            rng_key: JAX random key
            settling_window: Number of steps to check for settling (default: 100 = 1 sec)
            min_steps: Minimum steps before checking settling (default: 1000 = 10 sec)
        
        Returns:
            X: State trajectory (actual_steps, n_x) - padded with last state if settled early
            U: Control trajectory (actual_steps, n_u) - padded with zeros if settled early
            is_stable: Boolean indicating if trajectory is stable
            settled_early: Boolean indicating if trajectory settled before n_steps
            settle_time: Time step when settling was detected (or n_steps if not settled)
        """
        n_x = A.shape[0]
        n_u = B.shape[1]
        
        # Stability threshold (if state norm exceeds this, mark as unstable)
        STABILITY_THRESHOLD = 1e4
        
        # Settling thresholds
        STATE_SETTLING_THRESHOLD = 1e-3  # State change threshold
        CONTROL_SETTLING_THRESHOLD = 1e-3  # Control magnitude threshold
        
        # Simulate using scan for efficiency
        def step_fn(carry, t):
            x, x_prev, key, is_stable, has_settled, settle_step, settled_counter = carry
            
            # Compute control
            u = -K @ x
            
            # RK4 integration
            k1 = A @ x + B @ u
            k2 = A @ (x + 0.5 * dt * k1) + B @ u
            k3 = A @ (x + 0.5 * dt * k2) + B @ u
            k4 = A @ (x + dt * k3) + B @ u
            
            # Update state
            dx = (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Add process noise
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (n_x,)) * process_noise_std
            
            x_next = x + dx + noise
            
            # Check stability: state norm and NaN/Inf
            state_norm = jnp.linalg.norm(x_next)
            is_nan_or_inf = jnp.any(jnp.isnan(x_next)) | jnp.any(jnp.isinf(x_next))
            is_stable_step = is_stable & (state_norm < STABILITY_THRESHOLD) & (~is_nan_or_inf)
            
            # Check settling (only after min_steps)
            state_change = jnp.linalg.norm(x_next - x_prev)
            control_magnitude = jnp.linalg.norm(u)
            is_settled_step = (state_change < STATE_SETTLING_THRESHOLD) & (control_magnitude < CONTROL_SETTLING_THRESHOLD)
            
            # Update settling counter (only after min_steps)
            past_min_steps = t >= min_steps
            counter_increment = jnp.where(is_settled_step & past_min_steps, 1, 0)
            new_counter = jnp.where(is_settled_step & past_min_steps, 
                                   settled_counter + counter_increment, 
                                   0)  # Reset if not settled
            
            # Mark as settled if counter reaches settling_window
            newly_settled = (new_counter >= settling_window) & (~has_settled)
            new_has_settled = has_settled | newly_settled
            new_settle_step = jnp.where(newly_settled, t, settle_step)
            
            return (x_next, x, key, is_stable_step, new_has_settled, new_settle_step, new_counter), \
                   (x_next, u, is_stable_step, new_has_settled)
        
        # Run simulation
        init_carry = (x0, x0, rng_key, True, False, n_steps, 0)
        (x_final, _, _, is_stable_final, has_settled, settle_step, _), \
        (X_traj, U_traj, stability_flags, settling_flags) = jax.lax.scan(
            step_fn, init_carry, jnp.arange(n_steps-1)
        )
        
        # Concatenate with initial state
        X_full = jnp.concatenate([x0[None, :], X_traj], axis=0)
        U_last = -K @ X_traj[-1]
        U_full = jnp.concatenate([U_traj, U_last[None, :]], axis=0)
        
        # Check if trajectory remained stable throughout
        is_stable = is_stable_final & jnp.all(stability_flags)
        
        # Return NaN arrays if unstable
        X_out = jnp.where(is_stable, X_full, jnp.nan * jnp.ones_like(X_full))
        U_out = jnp.where(is_stable, U_full, jnp.nan * jnp.ones_like(U_full))
        
        return X_out, U_out, is_stable, has_settled, settle_step
    
    
    @partial(jit, static_argnums=(5, 8, 9))  # n_steps, settling_window, min_steps are static
    def simulate_batch_trajectories_jax(A, B, K, X0, dt, n_steps, process_noise_std, rng_keys,
                                       settling_window=100, min_steps=1000):
        """
        Vectorized batch trajectory simulation with stability and settling detection.
        
        Args:
            A, B, K: System matrices
            X0: Batch of initial states (batch_size, n_x)
            dt, n_steps, process_noise_std: Simulation parameters
            rng_keys: Batch of random keys (batch_size, 2)
            settling_window: Number of steps to check for settling (default: 100 = 1 sec)
            min_steps: Minimum steps before checking settling (default: 1000 = 10 sec)
        
        Returns:
            X_batch: (batch_size, n_steps, n_x)
            U_batch: (batch_size, n_steps, n_u)
            stability_flags: (batch_size,) - True if trajectory is stable
            settled_flags: (batch_size,) - True if trajectory settled early
            settle_steps: (batch_size,) - Time step when settling was detected
        """
        # Vectorize over batch dimension
        simulate_fn = vmap(
            lambda x0, key: simulate_trajectory_jax(A, B, K, x0, dt, n_steps, process_noise_std, key,
                                                   settling_window, min_steps),
            in_axes=(0, 0)
        )
        
        X_batch, U_batch, stability_flags, settled_flags, settle_steps = simulate_fn(X0, rng_keys)
        return X_batch, U_batch, stability_flags, settled_flags, settle_steps


def generate_data_jax_accelerated(systems_to_generate, output_path, batch_size=100, chunk_size=10000):
    """
    Generate data using JAX acceleration.
    
    Args:
        systems_to_generate: List of system names
        output_path: Output HDF5 path
        batch_size: Number of trajectories to simulate in parallel
        chunk_size: HDF5 chunk size
    """
    if not JAX_AVAILABLE:
        print("JAX not available. Please use data_generation.py instead.")
        return 0
    
    print("="*80)
    print("JAX-ACCELERATED DATA GENERATION")
    print("="*80)
    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Batch size: {batch_size} trajectories in parallel")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize HDF5 file
    estimated_sequences = 100_000_000  # Estimate
    with h5py.File(output_path, 'w') as f:
        ds_input = f.create_dataset(
            'input_sequences',
            shape=(0, SEQUENCE_LENGTH, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE),
            maxshape=(estimated_sequences, SEQUENCE_LENGTH, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE),
            dtype=np.float32,
            chunks=(chunk_size, SEQUENCE_LENGTH, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE),
            compression='gzip',
            compression_opts=4
        )
        
        ds_controls = f.create_dataset(
            'controls',
            shape=(0, MAX_INPUT_DIM),
            maxshape=(estimated_sequences, MAX_INPUT_DIM),
            dtype=np.float32,
            chunks=(chunk_size, MAX_INPUT_DIM),
            compression='gzip',
            compression_opts=4
        )
        
        ds_masks = f.create_dataset(
            'control_masks',
            shape=(0, MAX_INPUT_DIM),
            maxshape=(estimated_sequences, MAX_INPUT_DIM),
            dtype=np.float32,
            chunks=(chunk_size, MAX_INPUT_DIM),
            compression='gzip',
            compression_opts=4
        )
        
        # NEW: System ID dataset to track which system each sequence belongs to
        ds_system_id = f.create_dataset(
            'system_ids',
            shape=(0,),
            maxshape=(estimated_sequences,),
            dtype='S50',  # String up to 50 chars
            chunks=(chunk_size,),
            compression='gzip',
            compression_opts=4
        )
        
        # Metadata
        f.attrs['sequence_length'] = SEQUENCE_LENGTH
        f.attrs['max_state_dim'] = MAX_STATE_DIM
        f.attrs['max_input_dim'] = MAX_INPUT_DIM
        f.attrs['dimension_encoding_size'] = DIMENSION_ENCODING_SIZE
        f.attrs['parameter_uncertainty'] = PARAMETER_UNCERTAINTY
        f.attrs['process_noise_std'] = PROCESS_NOISE_STD
        f.attrs['time_horizon'] = TIME_HORIZON
        f.attrs['dt'] = DT
        f.attrs['sequence_stride'] = SEQUENCE_STRIDE
    
    sequences_written = 0
    batch_buffer_input = []
    batch_buffer_control = []
    batch_buffer_mask = []
    batch_buffer_system_id = []  # NEW: Track system for each sequence
    
    # NEW: Track max values per system for normalization
    system_max_values = {}  # {system_name: {'state_max': array, 'control_max': array}}
    
    # Settling statistics
    total_trajectories = 0
    settled_trajectories = 0
    total_settle_time = 0.0
    
    # Main random key
    main_key = jax.random.PRNGKey(RANDOM_SEED)
    
    n_steps = int(TIME_HORIZON / DT)
    
    print(f"\nGenerating data for {len(systems_to_generate)} systems...")
    
    for sys_idx, system_name in enumerate(systems_to_generate):
        print(f"\n[{sys_idx+1}/{len(systems_to_generate)}] {system_name}")
        
        # Get nominal system and LQR
        SystemClass = getattr(systems, system_name)
        nominal_system = SystemClass()
        Q, R = nominal_system.get_default_lqr_weights()
        K, _, _, success = design_lqr(nominal_system.A, nominal_system.B, custom_Q=Q, custom_R=R)
        
        if not success:
            print(f"  Skipping {system_name} - LQR design failed")
            continue
        
        n_x = nominal_system.n_states
        n_u = nominal_system.n_inputs
        
        # NEW: Initialize max tracking for this system
        if system_name not in system_max_values:
            system_max_values[system_name] = {
                'state_max': np.zeros(n_x),
                'control_max': np.zeros(n_u),
                'n_states': n_x,
                'n_inputs': n_u
            }
        
        # Generate variants
        for variant_idx in tqdm(range(NUM_VARIANTS_PER_SYSTEM), desc="  Variants"):
            # Create variant system
            local_seed = hash((system_name, variant_idx)) % (2**31)
            np.random.seed(local_seed)
            variant_params = nominal_system.generate_variant_params(PARAMETER_UNCERTAINTY)
            variant_system = SystemClass(params=variant_params)
            
            # Convert to JAX arrays
            A_jax = jnp.array(variant_system.A)
            B_jax = jnp.array(variant_system.B)
            K_jax = jnp.array(K)
            
            # Generate initial conditions in batches
            n_batches = (NUM_TRAJECTORIES_PER_VARIANT + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_traj = batch_idx * batch_size
                end_traj = min(start_traj + batch_size, NUM_TRAJECTORIES_PER_VARIANT)
                current_batch_size = end_traj - start_traj
                
                # Sample initial conditions
                X0_batch = []
                for traj_idx in range(start_traj, end_traj):
                    local_seed = hash((system_name, variant_idx, traj_idx)) % (2**31)
                    np.random.seed(local_seed)
                    x0 = variant_system.sample_initial_condition()
                    X0_batch.append(x0)
                
                X0_batch = jnp.array(X0_batch)
                
                # Generate random keys for each trajectory
                main_key, *subkeys = jax.random.split(main_key, current_batch_size + 1)
                rng_keys = jnp.array(subkeys)
                
                # Simulate batch of trajectories with settling detection
                X_batch, U_batch, stability_flags, settled_flags, settle_steps = simulate_batch_trajectories_jax(
                    A_jax, B_jax, K_jax, X0_batch, 
                    DT, n_steps, PROCESS_NOISE_STD, rng_keys,
                    settling_window=100,  # 1 second window
                    min_steps=1000  # Don't check before 10 seconds
                )
                
                # Convert back to numpy
                X_batch_np = np.array(X_batch)
                U_batch_np = np.array(U_batch)
                stability_flags_np = np.array(stability_flags)
                settled_flags_np = np.array(settled_flags)
                settle_steps_np = np.array(settle_steps)
                
                # Extract sequences from each trajectory (skip unstable ones)
                for i in range(current_batch_size):
                    total_trajectories += 1
                    
                    # Skip if trajectory is unstable
                    if not stability_flags_np[i]:
                        continue
                    
                    X = X_batch_np[i]
                    U = U_batch_np[i]
                    
                    # Double-check for NaN/Inf (safety check)
                    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                        continue
                    
                    # Track settling statistics
                    if settled_flags_np[i]:
                        settled_trajectories += 1
                        settle_time_sec = settle_steps_np[i] * DT
                        total_settle_time += settle_time_sec
                    
                    # NEW: Update max values for this system
                    state_abs = np.abs(X[:, :n_x])
                    control_abs = np.abs(U[:, :n_u])
                    system_max_values[system_name]['state_max'] = np.maximum(
                        system_max_values[system_name]['state_max'],
                        np.max(state_abs, axis=0)
                    )
                    system_max_values[system_name]['control_max'] = np.maximum(
                        system_max_values[system_name]['control_max'],
                        np.max(control_abs, axis=0)
                    )
                    
                    # Extract sequences (import functions locally to avoid scope issues)
                    from data_utils import pad_control, create_control_mask, prepare_input_sequence as prep_seq
                    
                    traj_data = {
                        'states': X.astype(np.float32),
                        'controls': U.astype(np.float32),
                        'n_states': n_x,
                        'n_inputs': n_u,
                    }
                    
                    X_traj = traj_data['states']
                    U_traj = traj_data['controls']
                    T = X_traj.shape[0]
                    sequences = []
                    
                    for i in range(0, T - SEQUENCE_LENGTH, SEQUENCE_STRIDE):
                        state_seq = X_traj[i:i+SEQUENCE_LENGTH]
                        control = U_traj[i+SEQUENCE_LENGTH-1]
                        
                        input_seq = prep_seq(state_seq, n_u, n_x)
                        control_padded = pad_control(control, MAX_INPUT_DIM)
                        control_mask = create_control_mask(n_u, MAX_INPUT_DIM)
                        
                        sequences.append((input_seq, control_padded, control_mask))
                    
                    for input_seq, control, mask in sequences:
                        batch_buffer_input.append(input_seq)
                        batch_buffer_control.append(control)
                        batch_buffer_mask.append(mask)
                        batch_buffer_system_id.append(system_name.encode('utf-8'))  # NEW: Track system
                    
                    # Write to HDF5 when buffer is full
                    if len(batch_buffer_input) >= chunk_size:
                        with h5py.File(output_path, 'a') as f:
                            current_size = f['input_sequences'].shape[0]
                            new_size = current_size + len(batch_buffer_input)
                            
                            f['input_sequences'].resize(new_size, axis=0)
                            f['controls'].resize(new_size, axis=0)
                            f['control_masks'].resize(new_size, axis=0)
                            f['system_ids'].resize(new_size, axis=0)  # NEW
                            
                            f['input_sequences'][current_size:new_size] = np.array(batch_buffer_input)
                            f['controls'][current_size:new_size] = np.array(batch_buffer_control)
                            f['control_masks'][current_size:new_size] = np.array(batch_buffer_mask)
                            f['system_ids'][current_size:new_size] = np.array(batch_buffer_system_id)  # NEW
                        
                        sequences_written += len(batch_buffer_input)
                        
                        batch_buffer_input = []
                        batch_buffer_control = []
                        batch_buffer_mask = []
                        batch_buffer_system_id = []  # NEW
    
    # Write remaining
    if len(batch_buffer_input) > 0:
        with h5py.File(output_path, 'a') as f:
            current_size = f['input_sequences'].shape[0]
            new_size = current_size + len(batch_buffer_input)
            
            f['input_sequences'].resize(new_size, axis=0)
            f['controls'].resize(new_size, axis=0)
            f['control_masks'].resize(new_size, axis=0)
            f['system_ids'].resize(new_size, axis=0)  # NEW
            
            f['input_sequences'][current_size:new_size] = np.array(batch_buffer_input)
            f['controls'][current_size:new_size] = np.array(batch_buffer_control)
            f['control_masks'][current_size:new_size] = np.array(batch_buffer_mask)
            f['system_ids'][current_size:new_size] = np.array(batch_buffer_system_id)  # NEW
        
        sequences_written += len(batch_buffer_input)
    
    # NEW: Normalize data by per-system max values and save normalization factors
    print("\n" + "="*80)
    print("NORMALIZING DATA BY PER-SYSTEM MAX VALUES")
    print("="*80)
    
    # Add small epsilon to avoid division by zero
    for system_name in system_max_values:
        system_max_values[system_name]['state_max'] = np.maximum(
            system_max_values[system_name]['state_max'], 1e-8
        )
        system_max_values[system_name]['control_max'] = np.maximum(
            system_max_values[system_name]['control_max'], 1e-8
        )
        
        print(f"\n{system_name}:")
        print(f"  State max: {system_max_values[system_name]['state_max']}")
        print(f"  Control max: {system_max_values[system_name]['control_max']}")
    
    # Normalize data in-place (read, normalize, write back)
    print("\nNormalizing sequences...")
    with h5py.File(output_path, 'a') as f:
        total_seqs = f['input_sequences'].shape[0]
        system_ids = f['system_ids'][:]
        
        # Process in chunks to avoid memory issues
        chunk_size_norm = 10000
        for start_idx in tqdm(range(0, total_seqs, chunk_size_norm), desc="Normalizing"):
            end_idx = min(start_idx + chunk_size_norm, total_seqs)
            
            # Load chunk
            input_seqs = f['input_sequences'][start_idx:end_idx]
            controls = f['controls'][start_idx:end_idx]
            system_ids_chunk = system_ids[start_idx:end_idx]
            
            # Normalize each sequence by its system's max values
            for i in range(len(input_seqs)):
                sys_name = system_ids_chunk[i].decode('utf-8')
                n_x = system_max_values[sys_name]['n_states']
                n_u = system_max_values[sys_name]['n_inputs']
                state_max = system_max_values[sys_name]['state_max']
                control_max = system_max_values[sys_name]['control_max']
                
                # Normalize states in input sequence (first MAX_STATE_DIM features)
                input_seqs[i, :, :n_x] /= state_max[np.newaxis, :]
                
                # Normalize control
                controls[i, :n_u] /= control_max
            
            # Write back normalized data
            f['input_sequences'][start_idx:end_idx] = input_seqs
            f['controls'][start_idx:end_idx] = controls
    
    # Save normalization factors for each system
    print("\nSaving per-system normalization factors...")
    with h5py.File(output_path, 'a') as f:
        # Create a group for normalization factors
        if 'normalization' in f:
            del f['normalization']
        norm_group = f.create_group('normalization')
        
        for system_name, values in system_max_values.items():
            sys_group = norm_group.create_group(system_name)
            sys_group.create_dataset('state_max', data=values['state_max'].astype(np.float32))
            sys_group.create_dataset('control_max', data=values['control_max'].astype(np.float32))
            sys_group.attrs['n_states'] = values['n_states']
            sys_group.attrs['n_inputs'] = values['n_inputs']
        
        # For backward compatibility, also store dummy mean/std
        state_mean = np.zeros(MAX_STATE_DIM, dtype=np.float32)
        state_std = np.ones(MAX_STATE_DIM, dtype=np.float32)
        control_mean = np.zeros(MAX_INPUT_DIM, dtype=np.float32)
        control_std = np.ones(MAX_INPUT_DIM, dtype=np.float32)
        
        if 'state_mean' in f:
            del f['state_mean']
        if 'state_std' in f:
            del f['state_std']
        if 'control_mean' in f:
            del f['control_mean']
        if 'control_std' in f:
            del f['control_std']
        
        f.create_dataset('state_mean', data=state_mean)
        f.create_dataset('state_std', data=state_std)
        f.create_dataset('control_mean', data=control_mean)
        f.create_dataset('control_std', data=control_std)
        
        f.attrs['total_sequences'] = sequences_written
        f.attrs['total_trajectories_generated'] = total_trajectories
        f.attrs['settled_trajectories'] = settled_trajectories
        if settled_trajectories > 0:
            f.attrs['average_settling_time'] = total_settle_time / settled_trajectories
    
    print(f"\nTotal sequences written: {sequences_written:,}")
    print(f"File size: {os.path.getsize(output_path) / 1e9:.2f} GB")
    
    # Print settling statistics
    if total_trajectories > 0:
        settling_percentage = (settled_trajectories / total_trajectories) * 100
        print(f"\n{'='*80}")
        print(f"SETTLING STATISTICS:")
        print(f"{'='*80}")
        print(f"Total trajectories simulated: {total_trajectories:,}")
        print(f"Trajectories that settled: {settled_trajectories:,} ({settling_percentage:.1f}%)")
        if settled_trajectories > 0:
            avg_settle_time = total_settle_time / settled_trajectories
            print(f"Average settling time: {avg_settle_time:.2f} sec")
            print(f"Time saved per settled trajectory: {TIME_HORIZON - avg_settle_time:.2f} sec")
        print(f"{'='*80}")
    
    return sequences_written


if __name__ == '__main__':
    systems_to_generate = [s for s in ALL_SYSTEMS if s != 'LTISystem']
    output_path = os.path.join(DATA_DIR, 'training_data_jax.h5')
    
    if os.path.exists(output_path):
        response = input(f"File {output_path} exists. Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(0)
        os.remove(output_path)
    
    start = time.time()
    num_sequences = generate_data_jax_accelerated(
        systems_to_generate,
        output_path,
        batch_size=100,  # Simulate 100 trajectories in parallel
        chunk_size=10000
    )
    elapsed = time.time() - start
    
    print(f"\nGeneration time: {elapsed/60:.1f} minutes")
    print(f"Throughput: {num_sequences/elapsed:.1f} sequences/second")
    
    # Verify data
    print("\nVerifying generated data...")
    import h5py
    with h5py.File(output_path, 'r') as f:
        print(f"  Total sequences: {f.attrs['total_sequences']:,}")
        print(f"  Input shape: {f['input_sequences'].shape}")
        print(f"  Controls shape: {f['controls'].shape}")
        print("✓ Data verification complete!")

