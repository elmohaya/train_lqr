"""
Testing/Evaluation Script using JAX/Flax

Evaluates trained transformer on test systems and compares with LQR.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import numpy as np
import h5py
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    TRANSFORMER_CONFIG, DATA_DIR, MODEL_DIR, RANDOM_SEED,
    MAX_STATE_DIM, MAX_INPUT_DIM, DIMENSION_ENCODING_SIZE,
    SEQUENCE_LENGTH, DT
)
from train_jax import UniversalLQRTransformer
from lqr_controller import design_lqr, simulate_lqr_controlled_system
from data_utils import create_dimension_encoding
from systems import *


def load_trained_model(model_path):
    """Load trained JAX model parameters."""
    print(f"Loading model from: {model_path}")
    
    # Create model
    model = UniversalLQRTransformer(
        d_model=TRANSFORMER_CONFIG['d_model'],
        n_heads=TRANSFORMER_CONFIG['n_heads'],
        n_layers=TRANSFORMER_CONFIG['n_layers'],
        d_ff=TRANSFORMER_CONFIG['d_ff'],
        dropout=TRANSFORMER_CONFIG['dropout'],
        max_seq_len=TRANSFORMER_CONFIG['max_seq_len'],
        max_state_dim=MAX_STATE_DIM,
        max_control_dim=MAX_INPUT_DIM,
        dimension_encoding_size=DIMENSION_ENCODING_SIZE
    )
    
    # Load parameters
    with open(model_path, 'rb') as f:
        params = pickle.load(f)
    
    print("Model loaded successfully!")
    return model, params


def load_normalization_stats(h5_file):
    """Load normalization statistics from HDF5."""
    with h5py.File(h5_file, 'r') as hf:
        state_mean = np.array(hf['state_mean'])
        state_std = np.array(hf['state_std'])
        control_mean = np.array(hf['control_mean'])
        control_std = np.array(hf['control_std'])
    return state_mean, state_std, control_mean, control_std


def transformer_controller(state_history, model, params, n_inputs, n_states, 
                          state_mean, state_std, control_mean, control_std):
    """
    Use transformer to predict control given state history.
    
    Args:
        state_history: numpy array of shape (seq_len, n_states) - raw states
        model: JAX model
        params: Model parameters
        n_inputs: Number of control inputs for this system
        n_states: Number of states for this system
        state_mean, state_std: Normalization stats
        control_mean, control_std: Normalization stats
    
    Returns:
        control: numpy array of shape (n_inputs,)
    """
    # Normalize states
    states_norm = state_history.copy()
    states_norm[:, :n_states] = (states_norm[:, :n_states] - state_mean[:n_states]) / state_std[:n_states]
    
    # Pad states
    states_padded = np.zeros((SEQUENCE_LENGTH, MAX_STATE_DIM), dtype=np.float32)
    states_padded[:, :n_states] = states_norm
    
    # Create dimension encoding
    dim_encoding = create_dimension_encoding(n_inputs, n_states)
    if hasattr(dim_encoding, 'numpy'):
        dim_encoding = dim_encoding.numpy()
    dim_encoding_repeated = np.tile(dim_encoding, (SEQUENCE_LENGTH, 1))
    
    # Concatenate
    input_seq = np.concatenate([states_padded, dim_encoding_repeated], axis=1)
    input_seq = jnp.array(input_seq[None, ...])  # Add batch dimension
    
    # Forward pass
    output = model.apply({'params': params}, input_seq, training=False)
    control_pred = np.array(output[0, -1, :])  # Last timestep, remove batch dim
    
    # Denormalize
    control_pred[:n_inputs] = control_pred[:n_inputs] * control_std[:n_inputs] + control_mean[:n_inputs]
    
    return control_pred[:n_inputs]


def simulate_transformer_controlled_system(system, model, params, x0, t_span, dt,
                                          state_mean, state_std, control_mean, control_std):
    """
    Simulate system using transformer controller.
    
    Args:
        system: LTI system
        model: JAX transformer model
        params: Model parameters
        x0: Initial state
        t_span: Total simulation time
        dt: Time step
        state_mean, state_std: Normalization stats
        control_mean, control_std: Normalization stats
    
    Returns:
        t: Time vector
        X: State trajectory
        U: Control trajectory
    """
    n_steps = int(t_span / dt)
    t = np.linspace(0, t_span, n_steps)
    
    n_states = system.n_states
    n_inputs = system.n_inputs
    
    X = np.zeros((n_steps, MAX_STATE_DIM))
    U = np.zeros((n_steps, MAX_INPUT_DIM))
    
    X[0, :n_states] = x0
    
    # Initialize state history
    state_history = np.zeros((SEQUENCE_LENGTH, MAX_STATE_DIM))
    state_history[0, :n_states] = x0
    
    for i in range(n_steps - 1):
        # Get control from transformer
        u = transformer_controller(
            state_history, model, params, n_inputs, n_states,
            state_mean, state_std, control_mean, control_std
        )
        
        U[i, :n_inputs] = u
        
        # Dynamics: x_dot = Ax + Bu
        x_current = X[i, :n_states]
        x_dot = system.A @ x_current + system.B @ u
        
        # Euler integration (simple for testing)
        x_next = x_current + dt * x_dot
        X[i+1, :n_states] = x_next
        
        # Update state history (rolling window)
        state_history = np.roll(state_history, -1, axis=0)
        state_history[-1, :n_states] = x_next
    
    # Last control
    U[-1] = U[-2]
    
    return t, X, U


def evaluate_system(system, model, params, state_mean, state_std, 
                   control_mean, control_std, test_duration=10.0):
    """
    Evaluate transformer vs LQR on a single system.
    
    Returns:
        results: Dictionary with metrics and trajectories
    """
    print(f"\nEvaluating: {system.name}")
    print(f"  States: {system.n_states}, Inputs: {system.n_inputs}")
    
    # Design LQR controller
    K, success = design_lqr(system)
    if not success:
        print(f"  [X] LQR design failed for {system.name}")
        return None
    
    # Sample initial condition
    x0 = system.sample_initial_condition() * 0.5  # Reduced for testing
    
    print(f"  Initial condition: ||x0|| = {np.linalg.norm(x0):.4f}")
    
    # Simulate with LQR
    print("  Simulating with LQR...")
    t_lqr, X_lqr, U_lqr = simulate_lqr_controlled_system(
        A=system.A, B=system.B, K=K, x0=x0,
        t_span=test_duration, dt=DT, process_noise_std=0.0
    )
    
    # Simulate with Transformer
    print("  Simulating with Transformer...")
    t_tf, X_tf, U_tf = simulate_transformer_controlled_system(
        system, model, params, x0, test_duration, DT,
        state_mean, state_std, control_mean, control_std
    )
    
    # Compute metrics
    n_states = system.n_states
    n_inputs = system.n_inputs
    
    # State costs
    state_cost_lqr = np.sum(np.linalg.norm(X_lqr[:, :n_states], axis=1)**2) * DT
    state_cost_tf = np.sum(np.linalg.norm(X_tf[:, :n_states], axis=1)**2) * DT
    
    # Control costs
    control_cost_lqr = np.sum(np.linalg.norm(U_lqr[:, :n_inputs], axis=1)**2) * DT
    control_cost_tf = np.sum(np.linalg.norm(U_tf[:, :n_inputs], axis=1)**2) * DT
    
    # Total costs
    total_cost_lqr = state_cost_lqr + 0.1 * control_cost_lqr
    total_cost_tf = state_cost_tf + 0.1 * control_cost_tf
    
    # Check stability (final state norm)
    final_state_norm_lqr = np.linalg.norm(X_lqr[-1, :n_states])
    final_state_norm_tf = np.linalg.norm(X_tf[-1, :n_states])
    
    print(f"  LQR:         State Cost = {state_cost_lqr:.4f}, Control Cost = {control_cost_lqr:.4f}, Total = {total_cost_lqr:.4f}")
    print(f"  Transformer: State Cost = {state_cost_tf:.4f}, Control Cost = {control_cost_tf:.4f}, Total = {total_cost_tf:.4f}")
    print(f"  Final state norm: LQR = {final_state_norm_lqr:.4f}, TF = {final_state_norm_tf:.4f}")
    
    performance_ratio = total_cost_tf / total_cost_lqr
    if performance_ratio < 1.2:
        verdict = "[OK] Excellent"
    elif performance_ratio < 2.0:
        verdict = "[OK] Good"
    elif performance_ratio < 5.0:
        verdict = "[WARNING] Acceptable"
    else:
        verdict = "[X] Poor"
    
    print(f"  Performance: {performance_ratio:.2f}x LQR cost {verdict}")
    
    results = {
        'system_name': system.name,
        'n_states': n_states,
        'n_inputs': n_inputs,
        'x0': x0,
        'lqr': {
            't': t_lqr,
            'X': X_lqr,
            'U': U_lqr,
            'state_cost': state_cost_lqr,
            'control_cost': control_cost_lqr,
            'total_cost': total_cost_lqr,
            'final_state_norm': final_state_norm_lqr
        },
        'transformer': {
            't': t_tf,
            'X': X_tf,
            'U': U_tf,
            'state_cost': state_cost_tf,
            'control_cost': control_cost_tf,
            'total_cost': total_cost_tf,
            'final_state_norm': final_state_norm_tf
        },
        'performance_ratio': performance_ratio,
        'verdict': verdict
    }
    
    return results


def plot_comparison(results, save_dir='./test_results'):
    """Plot comparison between LQR and Transformer."""
    os.makedirs(save_dir, exist_ok=True)
    
    system_name = results['system_name']
    n_states = results['n_states']
    n_inputs = results['n_inputs']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # States
    ax = axes[0]
    t_lqr = results['lqr']['t']
    X_lqr = results['lqr']['X'][:, :n_states]
    X_tf = results['transformer']['X'][:, :n_states]
    
    for i in range(min(n_states, 6)):  # Plot up to 6 states
        ax.plot(t_lqr, X_lqr[:, i], '-', label=f'LQR x{i+1}', alpha=0.7)
        ax.plot(t_lqr, X_tf[:, i], '--', label=f'TF x{i+1}', alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('States')
    ax.set_title(f'{system_name} - State Trajectories')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Controls
    ax = axes[1]
    U_lqr = results['lqr']['U'][:, :n_inputs]
    U_tf = results['transformer']['U'][:, :n_inputs]
    
    for i in range(n_inputs):
        ax.plot(t_lqr, U_lqr[:, i], '-', label=f'LQR u{i+1}', alpha=0.7)
        ax.plot(t_lqr, U_tf[:, i], '--', label=f'TF u{i+1}', alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Controls')
    ax.set_title(f'{system_name} - Control Inputs')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    safe_name = system_name.replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(save_dir, f'{safe_name}_comparison.png'), dpi=150)
    plt.close()
    
    print(f"  Plot saved: {safe_name}_comparison.png")


def main():
    print("="*70)
    print("JAX TRANSFORMER EVALUATION")
    print("="*70)
    
    # Load trained model
    model_path = os.path.join(MODEL_DIR, 'best_model_jax.pkl')
    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found: {model_path}")
        print("Please train the model first using: python train_jax.py")
        return
    
    model, params = load_trained_model(model_path)
    
    # Load normalization stats
    h5_file = os.path.join(DATA_DIR, 'training_data.h5')
    if not os.path.exists(h5_file):
        print(f"\nERROR: Training data not found: {h5_file}")
        print("Please generate data first using: python data_generation.py")
        return
    
    state_mean, state_std, control_mean, control_std = load_normalization_stats(h5_file)
    print("\nNormalization statistics loaded.")
    
    # Get test systems (sample from different categories)
    test_systems = [
        # Mechanical
        CartPole(),
        InvertedPendulum(),
        BallAndBeam(),
        
        # Robotics
        TwoLinkArm(),
        DifferentialDriveRobot(),
        
        # Vehicles
        VehicleLateralDynamics(),
        LongitudinalCruiseControl(),
        
        # Aerospace
        QuadrotorHover(),
        
        # Other
        DoubleIntegrator(),
        ChemicalReactor(),
    ]
    
    print(f"\nTesting on {len(test_systems)} systems...")
    print("="*70)
    
    # Evaluate each system
    all_results = []
    successful = 0
    failed = 0
    
    for system in test_systems:
        try:
            results = evaluate_system(
                system, model, params,
                state_mean, state_std, control_mean, control_std,
                test_duration=10.0
            )
            
            if results is not None:
                all_results.append(results)
                plot_comparison(results)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [X] Error evaluating {system.name}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Successful: {successful}/{len(test_systems)}")
    print(f"Failed: {failed}/{len(test_systems)}")
    
    if all_results:
        print("\nPerformance Summary:")
        print(f"{'System':<30} {'Ratio':>8} {'Verdict':<20}")
        print("-"*70)
        
        for res in all_results:
            print(f"{res['system_name']:<30} {res['performance_ratio']:>8.2f} {res['verdict']:<20}")
        
        # Overall statistics
        ratios = [r['performance_ratio'] for r in all_results]
        print("-"*70)
        print(f"{'AVERAGE':<30} {np.mean(ratios):>8.2f}")
        print(f"{'MEDIAN':<30} {np.median(ratios):>8.2f}")
        print(f"{'BEST':<30} {np.min(ratios):>8.2f}")
        print(f"{'WORST':<30} {np.max(ratios):>8.2f}")
        
        # Count verdicts
        excellent = sum(1 for r in all_results if 'Excellent' in r['verdict'])
        good = sum(1 for r in all_results if 'Good' in r['verdict'])
        acceptable = sum(1 for r in all_results if 'Acceptable' in r['verdict'])
        poor = sum(1 for r in all_results if 'Poor' in r['verdict'])
        
        print("\nVerdict Distribution:")
        print(f"  Excellent (<1.2x):  {excellent}")
        print(f"  Good (1.2-2.0x):    {good}")
        print(f"  Acceptable (2-5x):  {acceptable}")
        print(f"  Poor (>5x):         {poor}")
    
    print("\n" + "="*70)
    print(f"Results saved in: ./test_results/")
    print("="*70)


if __name__ == '__main__':
    main()

