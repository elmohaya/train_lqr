"""
Debug script to visualize LQR control for each system family.

This script creates ONE variant from each LTI system family,
applies LQR control, and plots the trajectories to understand
why we're getting large control signals.

Author: Debug investigation
Date: 2025-10-11
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import sys
import os
from pathlib import Path

# Add systems to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all system families
from systems.mechanical_systems import (
    MassSpringDamper, SimplePendulum, InvertedPendulum, 
    CartPole, DoublePendulum, Acrobot, FurutaPendulum,
    BallAndBeam, BallAndPlate, ReactionWheelPendulum,
    FlexibleBeam, MagneticLevitation, SuspensionSystem
)
from systems.robotics_systems import (
    TwoLinkArm, ThreeLinkManipulator, SixDOFManipulator,
    DifferentialDriveRobot, OmnidirectionalRobot, UnicycleRobot,
    FlexibleJointRobot, CableDrivenRobot, DualArmRobot,
    SCARARobot, PlanarBiped, SegwayRobot
)
from systems.aerospace_systems import (
    QuadrotorHover, FixedWingAircraft, VTOLLinearized
)
from systems.vehicle_systems import (
    VehicleLateralDynamics, LongitudinalCruiseControl,
    PlatooningModel
)
from systems.electrical_systems import (
    DCMotor, ACMotor
)
from systems.other_systems import (
    ChemicalReactor, LotkaVolterra, DoubleIntegrator
)

# ============================================================================
# CONFIGURATION
# ============================================================================
NOMINAL_A_B = True  # If True, use nominal matrices (NO uncertainty)
                    # If False, use ±10% uncertain matrices

# Simulation parameters
DT = 0.02  # 20ms sampling time
T_SIM = 10.0  # 10 seconds simulation
N_STEPS = int(T_SIM / DT)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "lqr_debug"
OUTPUT_DIR.mkdir(exist_ok=True)


def simulate_lqr_trajectory(system, x0, K, n_steps, dt, add_noise=False):
    """
    Simulate a trajectory under LQR control.
    
    Args:
        system: LTI system instance
        x0: Initial state
        K: LQR gain matrix
        n_steps: Number of simulation steps
        dt: Time step
        add_noise: Whether to add process noise
        
    Returns:
        X: State trajectory (n_steps, n_states)
        U: Control trajectory (n_steps, n_inputs)
        t: Time vector (n_steps,)
        stable: Whether trajectory remained stable
    """
    n_states = system.n_states
    n_inputs = system.n_inputs
    
    # Get system matrices (stored as attributes)
    A = system.A
    B = system.B
    
    # Initialize storage
    X = np.zeros((n_steps, n_states))
    U = np.zeros((n_steps, n_inputs))
    t = np.zeros(n_steps)
    
    # Set initial state
    X[0] = x0
    stable = True
    
    # Process noise std
    process_noise_std = 0.01 if add_noise else 0.0
    
    for i in range(n_steps - 1):
        # Compute control
        u = -K @ X[i]
        U[i] = u
        
        # Check for stability
        if np.any(np.isnan(u)) or np.any(np.isinf(u)) or np.linalg.norm(u) > 1e6:
            stable = False
            break
        
        # Dynamics: x_{k+1} = x_k + dt * (A x_k + B u_k) + noise
        dx = A @ X[i] + B @ u
        
        # Add process noise
        if add_noise:
            noise = np.random.randn(n_states) * process_noise_std
            dx += noise
        
        X[i+1] = X[i] + dt * dx
        t[i] = i * dt
        
        # Check state stability
        if np.any(np.isnan(X[i+1])) or np.any(np.isinf(X[i+1])) or np.linalg.norm(X[i+1]) > 1e6:
            stable = False
            break
    
    # Last control
    if stable:
        U[-1] = -K @ X[-1]
        t[-1] = (n_steps - 1) * dt
    
    return X, U, t, stable


def plot_system_debug(system_name, system, x0, K, X, U, t, stable, eigenvalues):
    """
    Create comprehensive debug plots for a system.
    
    Args:
        system_name: Name of the system
        system: System instance
        x0: Initial state
        K: LQR gain matrix
        X: State trajectory
        U: Control trajectory
        t: Time vector
        stable: Stability flag
        eigenvalues: Closed-loop eigenvalues
    """
    n_states = system.n_states
    n_inputs = system.n_inputs
    
    # Create figure with subplots (2x3 grid)
    fig = plt.figure(figsize=(18, 10))
    
    # 1. ALL STATE TRAJECTORIES ON ONE PLOT
    ax1 = plt.subplot(2, 3, 1)
    for i in range(n_states):
        ax1.plot(t, X[:, i], linewidth=2, label=f'x{i+1}', alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('State Values', fontsize=12)
    ax1.set_title('All State Trajectories', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', ncol=min(3, n_states))
    
    # 2. ALL CONTROL SIGNALS ON ONE PLOT
    ax2 = plt.subplot(2, 3, 2)
    for i in range(n_inputs):
        u_mean = np.mean(U[:, i])
        u_std = np.std(U[:, i])
        u_min = np.min(U[:, i])
        u_max = np.max(U[:, i])
        ax2.plot(t, U[:, i], linewidth=2, 
                label=f'u{i+1} (μ={u_mean:.1e}, max={u_max:.1e})', 
                alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Control Values', fontsize=12)
    ax2.set_title('All Control Signals', fontweight='bold', fontsize=14, color='red')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', ncol=min(2, n_inputs), fontsize=9)
    
    # 3. State norm over time
    ax3 = plt.subplot(2, 3, 3)
    state_norm = np.linalg.norm(X, axis=1)
    ax3.semilogy(t, state_norm, 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('||x(t)||', fontsize=12)
    ax3.set_title('State Norm (log scale)', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. Control norm over time
    ax4 = plt.subplot(2, 3, 4)
    control_norm = np.linalg.norm(U, axis=1)
    ax4.semilogy(t, control_norm, 'r-', linewidth=2)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('||u(t)||', fontsize=12)
    ax4.set_title('Control Norm (log scale)', fontweight='bold', fontsize=14, color='red')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    u_norm_mean = np.mean(control_norm)
    u_norm_max = np.max(control_norm)
    ax4.text(0.02, 0.98, 
            f'Mean: {u_norm_mean:.2e}\nMax: {u_norm_max:.2e}',
            transform=ax4.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 5. Eigenvalue plot
    ax5 = plt.subplot(2, 3, 5)
    
    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax5.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
    
    # Plot eigenvalues
    ax5.scatter(eigenvalues.real, eigenvalues.imag, s=100, c='blue', marker='x', linewidths=2)
    
    # Check stability
    max_eig_mag = np.max(np.abs(eigenvalues))
    is_stable = max_eig_mag < 1.0
    
    ax5.set_xlabel('Real', fontsize=12)
    ax5.set_ylabel('Imaginary', fontsize=12)
    ax5.set_title(f'Closed-Loop Eigenvalues\nMax |λ|={max_eig_mag:.4f}', fontweight='bold', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    ax5.legend()
    
    # Add stability indicator
    color = 'green' if is_stable else 'red'
    stability_text = 'STABLE' if is_stable else 'UNSTABLE'
    ax5.text(0.02, 0.98, stability_text,
            transform=ax5.transAxes, fontsize=12, fontweight='bold',
            color=color, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # 6. System info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Get Q and R matrices
    Q, R = system.get_default_lqr_weights()
    
    # Control statistics
    control_stats = []
    for i in range(n_inputs):
        u_i = U[:, i]
        control_stats.append({
            'mean': np.mean(u_i),
            'std': np.std(u_i),
            'min': np.min(u_i),
            'max': np.max(u_i)
        })
    
    info_text = f"""SYSTEM: {system_name}
{'='*40}
Config: {'NOMINAL (no uncertainty)' if NOMINAL_A_B else 'WITH ±10% uncertainty'}

States: {n_states}, Inputs: {n_inputs}

Q/R Ratios:
  Q max: {np.max(np.diag(Q)):.1f}
  R min: {np.min(np.diag(R)):.1f}
  Q/R ratio: {np.max(np.diag(Q))/np.min(np.diag(R)):.1f}

LQR Gain K:
  Max |K|: {np.max(np.abs(K)):.3e}

Control Statistics:"""
    
    for i, stats in enumerate(control_stats):
        info_text += f"\n  u[{i}]: max={stats['max']:.2e}, μ={stats['mean']:.2e}"
    
    info_text += f"\n\nStability: {'✓ STABLE' if stable else '✗ UNSTABLE'}"
    info_text += f"\nMax |λ|: {max_eig_mag:.4f}"
    
    ax6.text(0.05, 0.95, info_text, fontsize=10, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Overall title
    fig.suptitle(f'LQR Debug: {system_name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / f'{system_name}_debug.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    # Return summary
    return {
        'system': system_name,
        'n_states': n_states,
        'n_inputs': n_inputs,
        'stable': stable,
        'max_eig': np.max(np.abs(eigenvalues)),
        'control_mean': np.mean(np.abs(U)),
        'control_max': np.max(np.abs(U)),
        'control_std': np.std(U)
    }


def main():
    """Main debug function."""
    
    print("="*80)
    print("LQR DATA GENERATION DEBUG".center(80))
    print("="*80)
    print()
    print("Creating ONE variant from each system family...")
    if NOMINAL_A_B:
        print("🔵 MODE: NOMINAL matrices (NO parameter uncertainty)")
    else:
        print("🔴 MODE: UNCERTAIN matrices (±10% parameter variation)")
    print("Simulating LQR control with NO PROCESS NOISE")
    print(f"Simulation time: {T_SIM}s, dt: {DT}s, steps: {N_STEPS}")
    print()
    
    # Get all system classes
    all_systems = [
        # Mechanical
        MassSpringDamper, SimplePendulum, InvertedPendulum, 
        CartPole, DoublePendulum, Acrobot, FurutaPendulum,
        BallAndBeam, BallAndPlate, ReactionWheelPendulum,
        FlexibleBeam, MagneticLevitation, SuspensionSystem,
        # Robotics
        TwoLinkArm, ThreeLinkManipulator, SixDOFManipulator,
        DifferentialDriveRobot, OmnidirectionalRobot, UnicycleRobot,
        FlexibleJointRobot, CableDrivenRobot, DualArmRobot,
        SCARARobot, PlanarBiped, SegwayRobot,
        # Aerospace
        QuadrotorHover, FixedWingAircraft, VTOLLinearized,
        # Vehicle
        VehicleLateralDynamics, LongitudinalCruiseControl,
        PlatooningModel,
        # Electrical
        DCMotor,
        # Other
        ChemicalReactor, LotkaVolterra, DoubleIntegrator
    ]
    
    summaries = []
    
    for i, SystemClass in enumerate(all_systems, 1):
        system_name = SystemClass.__name__
        
        print(f"[{i}/{len(all_systems)}] Processing {system_name}...")
        
        try:
            # Create nominal system (no uncertainty)
            system = SystemClass()
            
            # Get system matrices (stored as attributes)
            A = system.A
            B = system.B
            Q, R = system.get_default_lqr_weights()
            
            # Design LQR controller
            from lqr_controller import compute_lqr_gain
            K, P, success = compute_lqr_gain(A, B, Q, R)
            
            if not success:
                raise RuntimeError(f"LQR design failed for {system_name}")
            
            # Get closed-loop eigenvalues
            A_cl = A - B @ K
            eigenvalues = np.linalg.eigvals(A_cl)
            
            # Generate random initial condition
            np.random.seed(42)  # Fixed seed for reproducibility
            x0 = system.sample_initial_condition()
            
            # Simulate trajectory
            X, U, t, stable = simulate_lqr_trajectory(
                system, x0, K, N_STEPS, DT, add_noise=False
            )
            
            # Plot and save
            summary = plot_system_debug(
                system_name, system, x0, K, X, U, t, stable, eigenvalues
            )
            summaries.append(summary)
            
            # Print summary
            status = "✓" if stable else "✗"
            print(f"  {status} Stable: {stable}, Max|eig|: {summary['max_eig']:.4f}, "
                  f"Max|u|: {summary['control_max']:.2e}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            summaries.append({
                'system': system_name,
                'stable': False,
                'error': str(e)
            })
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    # Count stable/unstable
    n_stable = sum(1 for s in summaries if s.get('stable', False))
    n_unstable = len(summaries) - n_stable
    
    print(f"Total systems: {len(summaries)}")
    print(f"Stable:        {n_stable}")
    print(f"Unstable:      {n_unstable}")
    print()
    
    # Find systems with large control signals
    print("Systems with large control signals (|u| > 100):")
    large_control_systems = [s for s in summaries if s.get('control_max', 0) > 100]
    
    if large_control_systems:
        for s in large_control_systems:
            print(f"  - {s['system']}: Max|u| = {s['control_max']:.2e}")
    else:
        print("  None! All control signals are reasonable.")
    
    print()
    print("="*80)
    print(f"All plots saved in: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

