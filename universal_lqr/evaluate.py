"""
Evaluation Script for Universal LQR Transformer
Test the trained transformer on new unseen systems
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from config import SEQUENCE_LENGTH, MODEL_DIR, LOG_DIR, DT, TIME_HORIZON
from transformer_model import UniversalLQRTransformer
from lqr_controller import design_lqr
from systems import *


def load_trained_model(model_path=None):
    """
    Load trained transformer model.
    
    Args:
        model_path: Path to model checkpoint (default: best_model.pt)
    
    Returns:
        model: Loaded model
        stats: Normalization statistics
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'best_model.pt')
    
    # Load normalization statistics
    stats_path = os.path.join(MODEL_DIR, 'normalization_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Create model
    from config import TRANSFORMER_CONFIG
    model = UniversalLQRTransformer(
        max_state_dim=stats['max_state_dim'],
        max_control_dim=stats['max_control_dim'],
        d_model=TRANSFORMER_CONFIG['d_model'],
        n_heads=TRANSFORMER_CONFIG['n_heads'],
        n_layers=TRANSFORMER_CONFIG['n_layers'],
        d_ff=TRANSFORMER_CONFIG['d_ff'],
        dropout=0.0,  # No dropout during inference
        max_seq_len=TRANSFORMER_CONFIG['max_seq_len']
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model, stats


def normalize_state(state, stats):
    """Normalize state using training statistics."""
    state_norm = (state - stats['state_mean']) / stats['state_std']
    return state_norm


def denormalize_control(control_norm, stats):
    """Denormalize control using training statistics."""
    control = control_norm * stats['control_std'] + stats['control_mean']
    return control


def simulate_with_transformer(model, system, stats, t_span=10.0, dt=0.02, x0=None):
    """
    Simulate a system using the transformer controller.
    
    Args:
        model: Trained transformer model
        system: LTI system instance
        stats: Normalization statistics
        t_span: Simulation time
        dt: Time step
        x0: Initial state (if None, sample random)
    
    Returns:
        t: Time vector
        X: State trajectory
        U: Control trajectory
    """
    device = next(model.parameters()).device
    model.eval()
    
    if x0 is None:
        x0 = system.sample_initial_condition()
    
    # Time vector
    t = np.arange(0, t_span, dt)
    n_steps = len(t)
    
    # Initialize
    X = np.zeros((n_steps, system.n_states))
    U = np.zeros((n_steps, system.n_inputs))
    X[0] = x0
    
    # State history buffer for transformer
    state_history = np.zeros((SEQUENCE_LENGTH, stats['max_state_dim']))
    
    # Initialize buffer with initial state
    for i in range(SEQUENCE_LENGTH):
        state_padded = np.zeros(stats['max_state_dim'])
        state_padded[:system.n_states] = x0
        state_history[i] = normalize_state(state_padded, stats)
    
    with torch.no_grad():
        for i in range(n_steps - 1):
            # Get current state
            current_state = X[i]
            
            # Prepare state for transformer
            state_padded = np.zeros(stats['max_state_dim'])
            state_padded[:system.n_states] = current_state
            state_norm = normalize_state(state_padded, stats)
            
            # Update history
            state_history = np.roll(state_history, -1, axis=0)
            state_history[-1] = state_norm
            
            # Predict control
            state_tensor = torch.FloatTensor(state_history).unsqueeze(0).to(device)
            control_pred = model(state_tensor)
            control_norm = control_pred[0, -1, :].cpu().numpy()
            
            # Denormalize
            control = denormalize_control(control_norm, stats)
            u = control[:system.n_inputs]
            U[i] = u
            
            # Simulate dynamics
            dx = system.A @ current_state + system.B @ u
            X[i + 1] = X[i] + dx * dt
        
        # Final control
        U[-1] = U[-2]
    
    return t, X, U


def compare_lqr_vs_transformer(system, model, stats, t_span=10.0, dt=0.02, x0=None):
    """
    Compare LQR controller vs Transformer controller.
    
    Returns:
        results: Dictionary with comparison results
    """
    if x0 is None:
        x0 = system.sample_initial_condition()
    
    # LQR controller
    Q, R = system.get_default_lqr_weights()
    K, _, _, lqr_success = design_lqr(system.A, system.B, custom_Q=Q, custom_R=R)
    
    if not lqr_success:
        print(f"Warning: LQR design failed for {system.name}")
        return None
    
    # Simulate with LQR
    from lqr_controller import simulate_lqr_controlled_system
    t_lqr, X_lqr, U_lqr = simulate_lqr_controlled_system(
        system.A, system.B, K, x0, t_span, dt
    )
    
    # Simulate with Transformer
    t_trans, X_trans, U_trans = simulate_with_transformer(
        model, system, stats, t_span, dt, x0
    )
    
    # Compute metrics
    lqr_state_cost = np.sum(X_lqr**2)
    trans_state_cost = np.sum(X_trans**2)
    
    lqr_control_cost = np.sum(U_lqr**2)
    trans_control_cost = np.sum(U_trans**2)
    
    results = {
        't_lqr': t_lqr,
        'X_lqr': X_lqr,
        'U_lqr': U_lqr,
        't_trans': t_trans,
        'X_trans': X_trans,
        'U_trans': U_trans,
        'lqr_state_cost': lqr_state_cost,
        'trans_state_cost': trans_state_cost,
        'lqr_control_cost': lqr_control_cost,
        'trans_control_cost': trans_control_cost,
        'system_name': system.name
    }
    
    return results


def plot_comparison(results, save_path=None):
    """Plot comparison between LQR and Transformer."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot states
    ax = axes[0]
    for i in range(results['X_lqr'].shape[1]):
        ax.plot(results['t_lqr'], results['X_lqr'][:, i], '--', 
               label=f'LQR State {i+1}', alpha=0.7)
        ax.plot(results['t_trans'], results['X_trans'][:, i], '-',
               label=f'Transformer State {i+1}', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State')
    ax.set_title(f"{results['system_name']}: State Trajectories")
    ax.legend()
    ax.grid(True)
    
    # Plot controls
    ax = axes[1]
    for i in range(results['U_lqr'].shape[1]):
        ax.plot(results['t_lqr'], results['U_lqr'][:, i], '--',
               label=f'LQR Control {i+1}', alpha=0.7)
        ax.plot(results['t_trans'], results['U_trans'][:, i], '-',
               label=f'Transformer Control {i+1}', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control')
    ax.set_title('Control Inputs')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_on_test_systems():
    """
    Evaluate trained transformer on test systems.
    """
    # Load model
    print("Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats = load_trained_model()
    model = model.to(device)
    
    # Create test systems (some that were in training, some new)
    test_systems = [
        CartPole(),
        InvertedPendulum(),
        QuadrotorHover(),
        VehicleLateralDynamics(),
        MassSpringDamper(),
    ]
    
    print(f"\nEvaluating on {len(test_systems)} test systems...")
    
    # Create results directory
    results_dir = os.path.join(LOG_DIR, 'evaluation')
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    for system in test_systems:
        print(f"\nTesting: {system.name}")
        
        results = compare_lqr_vs_transformer(system, model, stats, t_span=10.0, dt=DT)
        
        if results is not None:
            all_results.append(results)
            
            # Print metrics
            print(f"  LQR State Cost: {results['lqr_state_cost']:.4f}")
            print(f"  Transformer State Cost: {results['trans_state_cost']:.4f}")
            print(f"  Ratio: {results['trans_state_cost']/results['lqr_state_cost']:.4f}")
            
            # Plot
            plot_path = os.path.join(results_dir, f"{system.name}_comparison.png")
            plot_comparison(results, plot_path)
    
    print(f"\nEvaluation complete! Results saved to: {results_dir}")


if __name__ == '__main__':
    evaluate_on_test_systems()

