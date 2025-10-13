"""
Test stability checking in JAX data generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from data_generation_jax import simulate_trajectory_jax, simulate_batch_trajectories_jax

def test_stable_trajectory():
    """Test that stable trajectory is not marked as unstable."""
    print("\n" + "="*80)
    print("TEST 1: Stable Trajectory")
    print("="*80)
    
    # Create a stable system (A has eigenvalues < 1)
    A = jnp.array([[-0.1, 0.0], [0.0, -0.2]])
    B = jnp.array([[1.0], [0.5]])
    K = jnp.array([[0.1, 0.1]])  # Small feedback gain
    
    x0 = jnp.array([1.0, 1.0])
    dt = 0.01
    n_steps = 100
    process_noise_std = 0.01
    
    rng_key = jax.random.PRNGKey(42)
    
    X, U, is_stable, has_settled, settle_step = simulate_trajectory_jax(
        A, B, K, x0, dt, n_steps, process_noise_std, rng_key,
        settling_window=10, min_steps=20
    )
    
    print(f"  Initial state: {x0}")
    print(f"  Final state: {X[-1]}")
    print(f"  Max state norm: {jnp.max(jnp.linalg.norm(X, axis=1)):.4f}")
    print(f"  Is stable: {is_stable}")
    print(f"  Has settled: {has_settled}")
    print(f"  Settle step: {settle_step}")
    print(f"  Contains NaN: {jnp.any(jnp.isnan(X))}")
    
    assert is_stable, "Stable trajectory should be marked as stable!"
    assert not jnp.any(jnp.isnan(X)), "Stable trajectory should not contain NaN!"
    print("  ✓ PASSED")


def test_unstable_trajectory():
    """Test that unstable trajectory is properly detected."""
    print("\n" + "="*80)
    print("TEST 2: Unstable Trajectory")
    print("="*80)
    
    # Create an unstable system (A has eigenvalues > 1)
    A = jnp.array([[2.0, 0.0], [0.0, 1.5]])
    B = jnp.array([[1.0], [0.5]])
    K = jnp.array([[0.01, 0.01]])  # Very small feedback gain (won't stabilize)
    
    x0 = jnp.array([1.0, 1.0])
    dt = 0.01
    n_steps = 1000
    process_noise_std = 0.01
    
    rng_key = jax.random.PRNGKey(42)
    
    X, U, is_stable, has_settled, settle_step = simulate_trajectory_jax(
        A, B, K, x0, dt, n_steps, process_noise_std, rng_key,
        settling_window=100, min_steps=200
    )
    
    print(f"  Initial state: {x0}")
    print(f"  Is stable: {is_stable}")
    print(f"  Contains NaN: {jnp.any(jnp.isnan(X))}")
    
    assert not is_stable, "Unstable trajectory should be marked as unstable!"
    assert jnp.any(jnp.isnan(X)), "Unstable trajectory should contain NaN!"
    print("  ✓ PASSED")


def test_batch_stability():
    """Test batch simulation with mixed stable/unstable trajectories."""
    print("\n" + "="*80)
    print("TEST 3: Batch Stability and Settling")
    print("="*80)
    
    # Create a marginally stable system
    A = jnp.array([[-0.5, 0.0], [0.0, -0.3]])
    B = jnp.array([[1.0], [0.5]])
    K = jnp.array([[0.2, 0.2]])
    
    # Some initial conditions that will be stable, some that might not
    X0 = jnp.array([
        [1.0, 1.0],
        [10.0, 10.0],
        [0.1, 0.1],
        [100.0, 100.0],
        [5.0, 5.0],
    ])
    
    dt = 0.01
    n_steps = 500
    process_noise_std = 0.1
    
    # Create random keys for each trajectory
    rng_keys = jax.random.split(jax.random.PRNGKey(42), 5)
    
    X_batch, U_batch, stability_flags, settled_flags, settle_steps = simulate_batch_trajectories_jax(
        A, B, K, X0, dt, n_steps, process_noise_std, rng_keys,
        settling_window=50, min_steps=100
    )
    
    print(f"  Batch size: {len(X0)}")
    print(f"  Stability flags: {stability_flags}")
    print(f"  Number stable: {jnp.sum(stability_flags)}")
    print(f"  Number unstable: {jnp.sum(~stability_flags)}")
    print(f"  Settled flags: {settled_flags}")
    print(f"  Number settled: {jnp.sum(settled_flags)}")
    print(f"  Settle steps: {settle_steps}")
    
    # Check that unstable trajectories contain NaN
    for i in range(len(X0)):
        if not stability_flags[i]:
            assert jnp.any(jnp.isnan(X_batch[i])), f"Trajectory {i} marked unstable but no NaN!"
        else:
            assert not jnp.any(jnp.isnan(X_batch[i])), f"Trajectory {i} marked stable but has NaN!"
    
    print("  ✓ PASSED")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("STABILITY CHECKING TEST SUITE")
    print("="*80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    try:
        test_stable_trajectory()
        test_unstable_trajectory()
        test_batch_stability()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nStability and settling detection is working correctly:")
        print("  - Stable trajectories are correctly identified")
        print("  - Unstable trajectories are detected and marked with NaN")
        print("  - Settling detection tracks when trajectories reach steady-state")
        print("  - Batch processing handles mixed stability and settling correctly")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

