"""
Data Generation Pipeline for Universal LQR Transformer
"""

import numpy as np
import os
import pickle
import h5py
from tqdm import tqdm
import sys

from config import (
    SAVE_NEW_DATA, NUM_TRAJECTORIES_PER_VARIANT, TIME_HORIZON, DT,
    NUM_VARIANTS_PER_SYSTEM, PARAMETER_UNCERTAINTY, PROCESS_NOISE_STD,
    DATA_DIR, RANDOM_SEED, SEQUENCE_LENGTH, MAX_STATE_DIM, MAX_INPUT_DIM,
    DIMENSION_ENCODING_SIZE
)
from lqr_controller import design_lqr, simulate_lqr_controlled_system
from data_utils import create_dimension_encoding
from systems import *


def get_all_system_classes():
    """
    Get all LTI system classes to generate data for.
    
    Returns:
        List of system classes
    """
    system_classes = [
        # Mechanical systems (13)
        MassSpringDamper, SimplePendulum, InvertedPendulum, DoublePendulum,
        CartPole, Acrobot, FurutaPendulum, BallAndBeam, BallAndPlate,
        ReactionWheelPendulum, FlexibleBeam, MagneticLevitation, SuspensionSystem,
        
        # Electrical systems (1 - kept only DCMotor for reasonable dynamics)
        # Removed: ACMotor, Converters, Inverter, RLC circuits (too fast, < 1s settling)
        DCMotor,
        
        # Robotics systems (12 - added 8 new diverse systems!)
        TwoLinkArm, ThreeLinkManipulator, UnicycleRobot, DifferentialDriveRobot,
        SCARARobot, SegwayRobot, OmnidirectionalRobot,
        CableDrivenRobot, FlexibleJointRobot, PlanarBiped, SixDOFManipulator, DualArmRobot,
        # Note: Removed PlanarQuadruped (LQR design fails - too complex)
        
        # Vehicle systems (3)
        VehicleLateralDynamics, LongitudinalCruiseControl, PlatooningModel,
        
        # Aerospace systems (3)
        QuadrotorHover, FixedWingAircraft, VTOLLinearized,
        
        # Other systems (3)
        DoubleIntegrator, LotkaVolterra, ChemicalReactor
    ]
    return system_classes


def generate_system_variants(system_class, num_variants):
    """
    Generate multiple variants of a system with different parameters.
    
    Args:
        system_class: LTI system class
        num_variants: Number of variants to generate
    
    Returns:
        List of system instances with different parameters
    """
    variants = []
    
    for i in range(num_variants):
        # Create nominal system
        nominal_system = system_class()
        
        # Generate variant parameters
        variant_params = nominal_system.generate_variant_params(PARAMETER_UNCERTAINTY)
        
        # Create variant system
        variant_system = system_class(params=variant_params)
        variants.append(variant_system)
    
    return variants


def design_and_verify_lqr(system, verbose=False):
    """
    Design LQR controller and verify stability.
    
    Args:
        system: LTI system instance
        verbose: Print information
    
    Returns:
        K: LQR gain matrix (None if failed)
        success: Boolean indicating success
    """
    Q, R = system.get_default_lqr_weights()
    K, Q_used, R_used, success = design_lqr(system.A, system.B, 
                                             Q_weight=1.0, R_weight=1.0,
                                             custom_Q=Q, custom_R=R)
    
    if success and verbose:
        print(f"  LQR controller designed successfully for {system.name}")
    elif not success and verbose:
        print(f"  LQR controller failed for {system.name}")
    
    return K, success


def generate_uncertain_system(system, uncertainty_level=PARAMETER_UNCERTAINTY):
    """
    Generate an uncertain version of the system for data generation.
    The LQR is designed for the nominal system, but data comes from uncertain system.
    
    Args:
        system: Nominal system
        uncertainty_level: Uncertainty level (e.g., 0.30 for ±30%)
    
    Returns:
        A_uncertain, B_uncertain: Uncertain system matrices
    """
    # Generate uncertain parameters
    uncertain_params = system.generate_variant_params(uncertainty_level)
    
    # Create uncertain system
    uncertain_system = system.__class__(params=uncertain_params)
    
    return uncertain_system.A, uncertain_system.B


def generate_trajectory(system, K, trajectory_idx=0):
    """
    Generate a single trajectory using LQR control.
    
    The LQR controller K is designed for the nominal system,
    but the trajectory is generated from an uncertain system with process noise.
    
    Args:
        system: Nominal system (used for LQR design)
        K: LQR gain matrix
        trajectory_idx: Trajectory index (for reproducibility)
    
    Returns:
        t: Time vector
        X: State trajectory
        U: Control trajectory
    """
    # Set seed for reproducibility
    np.random.seed(RANDOM_SEED + trajectory_idx)
    
    # Generate uncertain system matrices
    A_uncertain, B_uncertain = generate_uncertain_system(system)
    
    # Sample initial condition
    x0 = system.sample_initial_condition()
    
    # Get typical state magnitude for noise scaling
    state_magnitude = system.get_typical_state_magnitude()
    noise_std = PROCESS_NOISE_STD * state_magnitude
    
    # Simulate with uncertainty and noise
    t, X, U = simulate_lqr_controlled_system(
        A=system.A,  # Nominal system (for LQR)
        B=system.B,
        K=K,
        x0=x0,
        t_span=TIME_HORIZON,
        dt=DT,
        process_noise_std=noise_std,
        uncertain_A=A_uncertain,  # Use uncertain system for simulation
        uncertain_B=B_uncertain
    )
    
    # Validate trajectory: check for NaN or Inf values
    if not np.isfinite(X).all() or not np.isfinite(U).all():
        return None, None, None  # Invalid trajectory
    
    return t, X, U


def generate_dataset_for_system(system_class, variant_idx, num_trajectories, verbose=True):
    """
    Generate dataset for a single system variant.
    
    Args:
        system_class: LTI system class
        variant_idx: Variant index
        num_trajectories: Number of trajectories to generate
        verbose: Print progress
    
    Returns:
        data: Dictionary containing trajectories
    """
    # Create system variant
    nominal_system = system_class()
    variant_params = nominal_system.generate_variant_params(PARAMETER_UNCERTAINTY)
    system = system_class(params=variant_params)
    
    # Design LQR controller
    K, success = design_and_verify_lqr(system, verbose=verbose)
    
    if not success:
        print(f"  Warning: LQR failed for {system.name} variant {variant_idx}")
        return None
    
    # Generate trajectories
    trajectories = []
    failed_count = 0
    max_attempts = num_trajectories * 2  # Allow retries for failed trajectories
    attempt = 0
    
    while len(trajectories) < num_trajectories and attempt < max_attempts:
        t, X, U = generate_trajectory(system, K, 
                                      trajectory_idx=variant_idx*1000 + attempt)
        
        # Skip invalid trajectories (NaN/Inf)
        if t is None:
            failed_count += 1
            attempt += 1
            continue
        
        trajectories.append({
            'time': t,
            'states': X,
            'controls': U
        })
        attempt += 1
    
    if failed_count > 0 and verbose:
        print(f"  Note: Skipped {failed_count} unstable trajectories for {system.name} variant {variant_idx}")
    
    data = {
        'system_name': system.name,
        'variant_idx': variant_idx,
        'n_states': system.n_states,
        'n_inputs': system.n_inputs,
        'trajectories': trajectories,
        'lqr_gain': K,
        'system_params': system.params
    }
    
    return data


def generate_all_data(save_dir=DATA_DIR):
    """
    Generate data for all systems and variants.
    Saves sequences directly in transformer-ready format (HDF5).
    
    Args:
        save_dir: Directory to save data
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all system classes
    system_classes = get_all_system_classes()
    
    print("="*70)
    print("GENERATING TRANSFORMER-READY DATA")
    print("="*70)
    print(f"System families: {len(system_classes)}")
    print(f"Variants per family: {NUM_VARIANTS_PER_SYSTEM}")
    print(f"Trajectories per variant: {NUM_TRAJECTORIES_PER_VARIANT}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Time horizon: {TIME_HORIZON}s, dt: {DT}s")
    print(f"Parameter uncertainty: ±{PARAMETER_UNCERTAINTY*100}%")
    print(f"Process noise std: {PROCESS_NOISE_STD}")
    print()
    
    # First pass: count total sequences for HDF5 allocation
    print("Step 1/3: Counting total sequences...")
    total_sequences = 0
    system_metadata = []
    
    for system_class in tqdm(system_classes, desc="Counting"):
        for variant_idx in range(NUM_VARIANTS_PER_SYSTEM):
            # Estimate sequences per trajectory (approximate)
            steps_per_traj = int(TIME_HORIZON / DT)
            n_sequences_per_traj = max(0, steps_per_traj - SEQUENCE_LENGTH)
            estimated_sequences = NUM_TRAJECTORIES_PER_VARIANT * n_sequences_per_traj
            
            system_metadata.append({
                'system_class': system_class,
                'variant_idx': variant_idx,
                'estimated_sequences': estimated_sequences
            })
            total_sequences += estimated_sequences
    
    print(f"Estimated total sequences: {total_sequences:,}")
    print(f"Estimated file size: ~{total_sequences * SEQUENCE_LENGTH * (MAX_STATE_DIM + DIMENSION_ENCODING_SIZE) * 4 / (1024**3):.1f} GB")
    
    # Step 2: Compute normalization statistics on a sample
    print("\nStep 2/3: Computing normalization statistics...")
    print("(Generating sample trajectories...)")
    
    sample_states = []
    sample_controls = []
    n_sample_systems = min(10, len(system_metadata))
    
    for i in range(n_sample_systems):
        meta = system_metadata[i * len(system_metadata) // n_sample_systems]
        system = meta['system_class']()
        K, success = design_and_verify_lqr(system, verbose=False)
        
        if success:
            t, X, U = generate_trajectory(system, K, trajectory_idx=i)
            if t is not None:
                sample_states.append(X[:, :system.n_states])
                sample_controls.append(U[:, :system.n_inputs])
    
    # Compute global statistics
    state_mean = np.zeros(MAX_STATE_DIM)
    state_std = np.ones(MAX_STATE_DIM)
    control_mean = np.zeros(MAX_INPUT_DIM)
    control_std = np.ones(MAX_INPUT_DIM)
    
    for dim in range(MAX_STATE_DIM):
        all_vals = []
        for states in sample_states:
            if states.shape[1] > dim:
                all_vals.extend(states[:, dim])
        if len(all_vals) > 0:
            state_mean[dim] = np.mean(all_vals)
            state_std[dim] = np.std(all_vals) + 1e-8
    
    for dim in range(MAX_INPUT_DIM):
        all_vals = []
        for controls in sample_controls:
            if controls.shape[1] > dim:
                all_vals.extend(controls[:, dim])
        if len(all_vals) > 0:
            control_mean[dim] = np.mean(all_vals)
            control_std[dim] = np.std(all_vals) + 1e-8
    
    print("Statistics computed.")
    
    # Step 3: Generate and save all sequences
    print("\nStep 3/3: Generating and saving sequences...")
    
    h5_file = os.path.join(save_dir, 'training_data.h5')
    
    with h5py.File(h5_file, 'w') as hf:
        # Create datasets (with estimated size + 10% buffer)
        buffer_size = int(total_sequences * 1.1)
        
        input_sequences = hf.create_dataset(
            'input_sequences',
            shape=(buffer_size, SEQUENCE_LENGTH, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE),
            dtype='float32',
            chunks=(1024, SEQUENCE_LENGTH, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE),
            compression='lzf'
        )
        
        controls = hf.create_dataset(
            'controls',
            shape=(buffer_size, MAX_INPUT_DIM),
            dtype='float32',
            chunks=(1024, MAX_INPUT_DIM),
            compression='lzf'
        )
        
        control_masks = hf.create_dataset(
            'control_masks',
            shape=(buffer_size, MAX_INPUT_DIM),
            dtype='float32',
            chunks=(1024, MAX_INPUT_DIM),
            compression='lzf'
        )
        
        # Store normalization stats
        hf.create_dataset('state_mean', data=state_mean)
        hf.create_dataset('state_std', data=state_std)
        hf.create_dataset('control_mean', data=control_mean)
        hf.create_dataset('control_std', data=control_std)
        
        # Generate and save sequences
        seq_idx = 0
        batch_size = 10000
        input_batch, control_batch, mask_batch = [], [], []
        failed_systems = []
        
        for meta in tqdm(system_metadata, desc="Generating data"):
            system_class = meta['system_class']
            variant_idx = meta['variant_idx']
            
            # Create system variant
            nominal_system = system_class()
            variant_params = nominal_system.generate_variant_params(PARAMETER_UNCERTAINTY)
            system = system_class(params=variant_params)
            
            # Design LQR
            K, success = design_and_verify_lqr(system, verbose=False)
            if not success:
                failed_systems.append((system_class.__name__, variant_idx))
                continue
            
            # Create dimension encoding once
            dim_encoding = create_dimension_encoding(system.n_inputs, system.n_states)
            if hasattr(dim_encoding, 'numpy'):
                dim_encoding = dim_encoding.numpy()
            dim_encoding_repeated = np.tile(dim_encoding, (SEQUENCE_LENGTH, 1))
            
            # Control mask
            control_mask = np.zeros(MAX_INPUT_DIM, dtype=np.float32)
            control_mask[:system.n_inputs] = 1.0
            
            # Generate trajectories
            for traj_idx in range(NUM_TRAJECTORIES_PER_VARIANT):
                t, X, U = generate_trajectory(system, K, 
                                              trajectory_idx=variant_idx*1000 + traj_idx)
                
                if t is None:
                    continue  # Skip invalid trajectories
                
                # Extract all sequences from this trajectory
                traj_len = len(t)
                n_sequences = traj_len - SEQUENCE_LENGTH
                
                for seq_start in range(n_sequences):
                    # Extract and normalize states
                    states = X[seq_start:seq_start + SEQUENCE_LENGTH, :].copy()
                    states[:, :system.n_states] = (states[:, :system.n_states] - state_mean[:system.n_states]) / state_std[:system.n_states]
                    
                    # Pad states
                    states_padded = np.zeros((SEQUENCE_LENGTH, MAX_STATE_DIM), dtype=np.float32)
                    states_padded[:, :system.n_states] = states
                    
                    # Concatenate with dimension encoding
                    input_seq = np.concatenate([states_padded, dim_encoding_repeated], axis=1)
                    
                    # Extract and normalize control
                    control = U[seq_start + SEQUENCE_LENGTH, :].copy()
                    control[:system.n_inputs] = (control[:system.n_inputs] - control_mean[:system.n_inputs]) / control_std[:system.n_inputs]
                    
                    # Pad control
                    control_padded = np.zeros(MAX_INPUT_DIM, dtype=np.float32)
                    control_padded[:system.n_inputs] = control
                    
                    # Add to batch
                    input_batch.append(input_seq)
                    control_batch.append(control_padded)
                    mask_batch.append(control_mask)
                    
                    # Write batch when full
                    if len(input_batch) >= batch_size:
                        batch_end = seq_idx + len(input_batch)
                        input_sequences[seq_idx:batch_end] = np.array(input_batch)
                        controls[seq_idx:batch_end] = np.array(control_batch)
                        control_masks[seq_idx:batch_end] = np.array(mask_batch)
                        
                        seq_idx = batch_end
                        input_batch, control_batch, mask_batch = [], [], []
        
        # Write remaining batch
        if len(input_batch) > 0:
            batch_end = seq_idx + len(input_batch)
            input_sequences[seq_idx:batch_end] = np.array(input_batch)
            controls[seq_idx:batch_end] = np.array(control_batch)
            control_masks[seq_idx:batch_end] = np.array(mask_batch)
            seq_idx = batch_end
        
        # Resize datasets to actual size
        input_sequences.resize((seq_idx, SEQUENCE_LENGTH, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE))
        controls.resize((seq_idx, MAX_INPUT_DIM))
        control_masks.resize((seq_idx, MAX_INPUT_DIM))
        
        # Store metadata
        hf.attrs['total_sequences'] = seq_idx
        hf.attrs['sequence_length'] = SEQUENCE_LENGTH
        hf.attrs['max_state_dim'] = MAX_STATE_DIM
        hf.attrs['max_input_dim'] = MAX_INPUT_DIM
    
    print(f"\n{'='*70}")
    print("DATA GENERATION COMPLETE!")
    print(f"Total sequences saved: {seq_idx:,}")
    print(f"Failed systems: {len(failed_systems)}")
    print(f"File: {h5_file}")
    print(f"Size: {os.path.getsize(h5_file) / (1024**3):.2f} GB")
    print("="*70)
    
    if failed_systems:
        print("\nFailed systems:")
        for name, idx in failed_systems:
            print(f"  - {name}, variant {idx}")
    
    return seq_idx


def load_data(data_dir=DATA_DIR):
    """
    Load existing data from file.
    
    Args:
        data_dir: Directory containing data
    
    Returns:
        all_data: List of dataset dictionaries
    """
    data_file = os.path.join(data_dir, 'lqr_training_data.pkl')
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Generating new data...")
        return None
    
    print(f"Loading existing data from: {data_file}")
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"Loaded {len(all_data)} system variants")
    return all_data


def main():
    """
    Main function to generate or load data.
    """
    if SAVE_NEW_DATA:
        print("Generating new data...")
        all_data = generate_all_data()
    else:
        print("Attempting to load existing data...")
        all_data = load_data()
        
        if all_data is None:
            print("No existing data found. Generating new data...")
            all_data = generate_all_data()
    
    return all_data


if __name__ == '__main__':
    main()

