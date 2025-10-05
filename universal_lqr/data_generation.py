"""
Data Generation Pipeline for Universal LQR Transformer
"""

import numpy as np
import os
import pickle
from tqdm import tqdm
import sys

from config import (
    SAVE_NEW_DATA, NUM_TRAJECTORIES_PER_VARIANT, TIME_HORIZON, DT,
    NUM_VARIANTS_PER_SYSTEM, PARAMETER_UNCERTAINTY, PROCESS_NOISE_STD,
    DATA_DIR, RANDOM_SEED
)
from lqr_controller import design_lqr, simulate_lqr_controlled_system
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
        print(f"  ✓ LQR controller designed successfully for {system.name}")
    elif not success and verbose:
        print(f"  ✗ LQR controller failed for {system.name}")
    
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
    
    Args:
        save_dir: Directory to save data
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all system classes
    system_classes = get_all_system_classes()
    
    print(f"Generating data for {len(system_classes)} system families")
    print(f"  - {NUM_VARIANTS_PER_SYSTEM} variants per family")
    print(f"  - {NUM_TRAJECTORIES_PER_VARIANT} trajectories per variant")
    print(f"  - Time horizon: {TIME_HORIZON}s, dt: {DT}s")
    print(f"  - Parameter uncertainty: ±{PARAMETER_UNCERTAINTY*100}%")
    print(f"  - Process noise std: {PROCESS_NOISE_STD}")
    print()
    
    all_data = []
    failed_systems = []
    
    for system_class in tqdm(system_classes, desc="System families"):
        print(f"\nGenerating data for: {system_class.__name__}")
        
        for variant_idx in range(NUM_VARIANTS_PER_SYSTEM):
            data = generate_dataset_for_system(
                system_class, 
                variant_idx, 
                NUM_TRAJECTORIES_PER_VARIANT,
                verbose=(variant_idx == 0)  # Only print for first variant
            )
            
            if data is not None:
                all_data.append(data)
            else:
                failed_systems.append((system_class.__name__, variant_idx))
    
    # Save data
    data_file = os.path.join(save_dir, 'lqr_training_data.pkl')
    with open(data_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"\n{'='*60}")
    print(f"Data generation complete!")
    print(f"  - Total system variants: {len(all_data)}")
    print(f"  - Total trajectories: {len(all_data) * NUM_TRAJECTORIES_PER_VARIANT}")
    print(f"  - Failed systems: {len(failed_systems)}")
    print(f"  - Data saved to: {data_file}")
    print(f"{'='*60}")
    
    if failed_systems:
        print("\nFailed systems:")
        for name, idx in failed_systems:
            print(f"  - {name}, variant {idx}")
    
    return all_data


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

