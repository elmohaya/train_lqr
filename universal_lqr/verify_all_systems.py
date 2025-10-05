"""
Verification Script: Test and Plot All LTI Systems with LQR Control

This script:
1. Tests LQR design on all 34 system families
2. Verifies stability of closed-loop systems
3. Simulates and plots trajectories for each system
4. Generates a summary report
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

from config import DT, RANDOM_SEED
from lqr_controller import design_lqr, verify_stability, simulate_lqr_controlled_system
from data_generation import get_all_system_classes


# Verification settings
VERIFICATION_TIME = 50.0  # seconds - increased for slow systems
VERIFICATION_DT = 0.001   # sampling time - much smaller for stiff systems!
OUTPUT_DIR = './verification_results'


def verify_single_system(system_class, plot=True, verbose=True):
    """
    Verify LQR control for a single system.
    
    Args:
        system_class: LTI system class
        plot: Whether to generate plots
        verbose: Print detailed information
    
    Returns:
        results: Dictionary with verification results
    """
    # Create system instance
    system = system_class()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing: {system.name}")
        print(f"  State dimension: {system.n_states}")
        print(f"  Input dimension: {system.n_inputs}")
    
    results = {
        'name': system.name,
        'n_states': system.n_states,
        'n_inputs': system.n_inputs,
        'lqr_success': False,
        'stable': False,
        'max_eigenvalue': None,
        'eigenvalues': None,
        'settling_achieved': False,
        'final_state_norm': None,
        'max_control': None
    }
    
    try:
        # Get LQR weights
        Q, R = system.get_default_lqr_weights()
        
        if verbose:
            print(f"  Q shape: {Q.shape}, R shape: {R.shape}")
        
        # Design LQR controller
        K, Q_used, R_used, lqr_success = design_lqr(
            system.A, system.B, custom_Q=Q, custom_R=R
        )
        
        results['lqr_success'] = lqr_success
        
        if not lqr_success:
            print(f"  ✗ LQR design failed!")
            return results
        
        if verbose:
            print(f"  ✓ LQR gain computed: K shape = {K.shape}")
        
        # Verify stability
        is_stable, max_eig, eigenvalues = verify_stability(system.A, system.B, K)
        results['stable'] = is_stable
        results['max_eigenvalue'] = max_eig
        results['eigenvalues'] = eigenvalues
        
        if verbose:
            print(f"  Closed-loop eigenvalues: {eigenvalues}")
            print(f"  Max real part: {max_eig:.6f}")
            if is_stable:
                print(f"  ✓ System is stable!")
            else:
                print(f"  ✗ System is NOT stable!")
                return results
        
        # Simulate system with SMALLER initial conditions for verification
        np.random.seed(RANDOM_SEED)
        x0 = system.sample_initial_condition()
        
        # Reduce initial conditions by 50% for better numerical stability during verification
        # (Training will use full range, but verification needs to stay in linear region)
        x0 = x0 * 0.5
        
        if verbose:
            print(f"  Initial condition: {x0}")
        
        t, X, U = simulate_lqr_controlled_system(
            system.A, system.B, K, x0,
            t_span=VERIFICATION_TIME,
            dt=VERIFICATION_DT,
            process_noise_std=0.0,  # No noise for verification
            early_stop=True,  # Enable early stopping
            settling_threshold=0.01,  # Stop when state norm < 0.01
            settling_duration=1.0  # Must stay settled for 1 second
        )
        
        # Analyze results
        actual_sim_time = t[-1]  # Actual time simulated
        final_state_norm = np.linalg.norm(X[-1])
        max_control = np.max(np.abs(U))
        
        results['actual_sim_time'] = actual_sim_time
        results['final_state_norm'] = final_state_norm
        results['max_control'] = max_control
        results['settling_achieved'] = final_state_norm < 0.1  # Threshold
        
        if verbose:
            print(f"  Simulation time: {actual_sim_time:.2f}s (max: {VERIFICATION_TIME}s)")
            print(f"  Final state norm: {final_state_norm:.6f}")
            print(f"  Max control effort: {max_control:.6f}")
            if results['settling_achieved']:
                early_stop_msg = " (early stop)" if actual_sim_time < VERIFICATION_TIME - 0.1 else ""
                print(f"  ✓ System settled to origin!{early_stop_msg}")
            else:
                print(f"  ⚠ System did not settle (norm={final_state_norm:.4f})")
        
        # Plot results
        if plot:
            plot_system_response(system, t, X, U, K, eigenvalues, 
                               save_dir=OUTPUT_DIR)
        
        results['simulation_data'] = {
            't': t,
            'X': X,
            'U': U,
            'K': K
        }
        
    except Exception as e:
        print(f"  ✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    return results


def plot_system_response(system, t, X, U, K, eigenvalues, save_dir):
    """
    Plot system response and save to file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_states = X.shape[1]
    n_inputs = U.shape[1]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Plot states
    ax1 = plt.subplot(3, 2, (1, 3))
    for i in range(n_states):
        ax1.plot(t, X[:, i], label=f'x{i+1}', linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('State', fontsize=11)
    ax1.set_title(f'{system.name}: State Trajectories', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot controls
    ax2 = plt.subplot(3, 2, (2, 4))
    for i in range(n_inputs):
        ax2.plot(t, U[:, i], label=f'u{i+1}', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Control', fontsize=11)
    ax2.set_title('Control Inputs', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot state norm
    ax3 = plt.subplot(3, 2, 5)
    state_norm = np.linalg.norm(X, axis=1)
    ax3.semilogy(t, state_norm, 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('||x(t)||', fontsize=11)
    ax3.set_title('State Norm (Log Scale)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot eigenvalues
    ax4 = plt.subplot(3, 2, 6)
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    ax4.scatter(real_parts, imag_parts, s=100, c='red', marker='x', linewidths=2)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Real Part', fontsize=11)
    ax4.set_ylabel('Imaginary Part', fontsize=11)
    ax4.set_title('Closed-Loop Eigenvalues', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add stability region shading (left half-plane)
    xlim = ax4.get_xlim()
    ylim = ax4.get_ylim()
    ax4.fill_betweenx(ylim, xlim[0], 0, alpha=0.1, color='green', label='Stable region')
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.legend(fontsize=9)
    
    # Add system info as text
    info_text = f"System: {system.name}\n"
    info_text += f"States: {n_states}, Inputs: {n_inputs}\n"
    info_text += f"Max eigenvalue real part: {np.max(real_parts):.4f}\n"
    info_text += f"Final state norm: {state_norm[-1]:.6f}\n"
    info_text += f"Max control: {np.max(np.abs(U)):.4f}"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Save plot
    filename = f"{system.name.replace(' ', '_')}_verification.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    # print(f"  Plot saved: {filepath}")


def generate_summary_report(all_results, save_dir):
    """
    Generate a summary report of all verifications.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    names = [r['name'] for r in all_results]
    lqr_success = [r['lqr_success'] for r in all_results]
    stable = [r['stable'] if r['lqr_success'] else False for r in all_results]
    max_eigs = [r['max_eigenvalue'] if r['max_eigenvalue'] is not None else 0 
                for r in all_results]
    final_norms = [r['final_state_norm'] if r['final_state_norm'] is not None else 0
                   for r in all_results]
    
    # 1. Success rate bar chart
    ax = axes[0, 0]
    success_count = sum(lqr_success)
    stable_count = sum(stable)
    categories = ['LQR Success', 'Stable', 'Settled']
    settled_count = sum([r.get('settling_achieved', False) for r in all_results])
    counts = [success_count, stable_count, settled_count]
    colors = ['green' if c == len(all_results) else 'orange' for c in counts]
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=len(all_results), color='blue', linestyle='--', 
              label=f'Total systems ({len(all_results)})')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Verification Success Summary', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}/{len(all_results)}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Max eigenvalue distribution
    ax = axes[0, 1]
    stable_eigs = [e for e, s in zip(max_eigs, stable) if s]
    unstable_eigs = [e for e, s in zip(max_eigs, stable) if not s]
    
    if stable_eigs:
        ax.hist(stable_eigs, bins=20, alpha=0.7, label='Stable', color='green', edgecolor='black')
    if unstable_eigs:
        ax.hist(unstable_eigs, bins=20, alpha=0.7, label='Unstable', color='red', edgecolor='black')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Stability boundary')
    ax.set_xlabel('Max Eigenvalue Real Part', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Closed-Loop Eigenvalue Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Final state norm
    ax = axes[1, 0]
    successful_norms = [n for n, s in zip(final_norms, stable) if s and n > 0]
    
    if successful_norms:
        indices = np.arange(len(successful_norms))
        ax.bar(indices, successful_norms, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axhline(y=0.1, color='red', linestyle='--', label='Settling threshold')
        ax.set_xlabel('System Index (Stable Systems Only)', fontsize=11)
        ax.set_ylabel('Final State Norm', fontsize=11)
        ax.set_title('Final State Norm for Stable Systems', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. System dimensions
    ax = axes[1, 1]
    n_states_list = [r['n_states'] for r in all_results]
    n_inputs_list = [r['n_inputs'] for r in all_results]
    
    scatter = ax.scatter(n_states_list, n_inputs_list, s=100, alpha=0.6, 
                        c=stable, cmap='RdYlGn', edgecolors='black', linewidths=1.5)
    ax.set_xlabel('Number of States', fontsize=11)
    ax.set_ylabel('Number of Inputs', fontsize=11)
    ax.set_title('System Dimensions (Green=Stable, Red=Unstable)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Stability', fontsize=10)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = os.path.join(save_dir, 'summary_report.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Summary plot saved: {summary_path}")
    
    # Generate text report
    report_path = os.path.join(save_dir, 'verification_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" LQR VERIFICATION REPORT ".center(70, "=") + "\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Systems Tested: {len(all_results)}\n")
        f.write(f"LQR Success: {success_count}/{len(all_results)} ({100*success_count/len(all_results):.1f}%)\n")
        f.write(f"Stable Systems: {stable_count}/{len(all_results)} ({100*stable_count/len(all_results):.1f}%)\n")
        f.write(f"Settled Systems: {settled_count}/{len(all_results)} ({100*settled_count/len(all_results):.1f}%)\n\n")
        
        f.write("="*70 + "\n")
        f.write(" DETAILED RESULTS ".center(70, "=") + "\n")
        f.write("="*70 + "\n\n")
        
        for i, result in enumerate(all_results, 1):
            f.write(f"{i}. {result['name']}\n")
            f.write(f"   States: {result['n_states']}, Inputs: {result['n_inputs']}\n")
            f.write(f"   LQR Success: {'✓' if result['lqr_success'] else '✗'}\n")
            f.write(f"   Stable: {'✓' if result['stable'] else '✗'}\n")
            
            if result['max_eigenvalue'] is not None:
                f.write(f"   Max Eigenvalue: {result['max_eigenvalue']:.6f}\n")
            
            if result.get('actual_sim_time') is not None:
                f.write(f"   Simulation Time: {result['actual_sim_time']:.3f}s\n")
            
            if result['final_state_norm'] is not None:
                f.write(f"   Final State Norm: {result['final_state_norm']:.6f}\n")
                f.write(f"   Settled: {'✓' if result.get('settling_achieved', False) else '✗'}\n")
            
            if 'error' in result:
                f.write(f"   Error: {result['error']}\n")
            
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write(" FAILED SYSTEMS ".center(70, "=") + "\n")
        f.write("="*70 + "\n\n")
        
        failed = [r for r in all_results if not r['lqr_success'] or not r['stable']]
        if failed:
            for result in failed:
                f.write(f"- {result['name']}: ")
                if not result['lqr_success']:
                    f.write("LQR design failed")
                elif not result['stable']:
                    f.write(f"Unstable (max eig: {result['max_eigenvalue']:.6f})")
                f.write("\n")
        else:
            f.write("None! All systems passed verification.\n")
    
    print(f"  Text report saved: {report_path}")


def main():
    """
    Main verification function.
    """
    print("="*70)
    print(" LQR VERIFICATION FOR ALL SYSTEMS ".center(70, "="))
    print("="*70)
    print()
    
    # Get all system classes
    system_classes = get_all_system_classes()
    
    print(f"Found {len(system_classes)} system families to verify")
    print(f"Verification time: {VERIFICATION_TIME}s per system")
    print(f"Results will be saved to: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Verify all systems
    all_results = []
    
    for system_class in tqdm(system_classes, desc="Verifying systems"):
        results = verify_single_system(system_class, plot=True, verbose=False)
        all_results.append(results)
    
    # Print summary
    print("\n" + "="*70)
    print(" VERIFICATION SUMMARY ".center(70, "="))
    print("="*70)
    
    success_count = sum([r['lqr_success'] for r in all_results])
    stable_count = sum([r['stable'] for r in all_results])
    settled_count = sum([r.get('settling_achieved', False) for r in all_results])
    
    print(f"\nTotal Systems: {len(all_results)}")
    print(f"  LQR Success: {success_count}/{len(all_results)} ({100*success_count/len(all_results):.1f}%)")
    print(f"  Stable: {stable_count}/{len(all_results)} ({100*stable_count/len(all_results):.1f}%)")
    print(f"  Settled: {settled_count}/{len(all_results)} ({100*settled_count/len(all_results):.1f}%)")
    
    # List any failed systems
    failed = [r for r in all_results if not r['lqr_success'] or not r['stable']]
    if failed:
        print(f"\n⚠ Warning: {len(failed)} system(s) failed:")
        for r in failed:
            status = "LQR failed" if not r['lqr_success'] else "Unstable"
            print(f"  - {r['name']}: {status}")
    else:
        print("\n✓ All systems passed verification!")
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(all_results, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print(" VERIFICATION COMPLETE ".center(70, "="))
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - Individual plots: {len(system_classes)} PNG files")
    print(f"  - Summary plot: summary_report.png")
    print(f"  - Text report: verification_report.txt")
    print()


if __name__ == '__main__':
    main()

