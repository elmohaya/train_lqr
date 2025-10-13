#!/usr/bin/env python3
"""
Test script for JAX systems and LQR controller
Verifies that all systems work correctly with JAX
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np

from systems import __all__ as ALL_SYSTEMS
import systems
from lqr_controller import design_lqr, simulate_lqr_controlled_system


def test_system(system_name):
    """Test a single system"""
    try:
        # Create system
        SystemClass = getattr(systems, system_name)
        system = SystemClass()
        
        # Check matrices are JAX arrays
        assert isinstance(system.A, jnp.ndarray), f"{system_name}: A is not JAX array"
        assert isinstance(system.B, jnp.ndarray), f"{system_name}: B is not JAX array"
        
        # Get LQR weights
        Q, R = system.get_default_lqr_weights()
        assert isinstance(Q, jnp.ndarray), f"{system_name}: Q is not JAX array"
        assert isinstance(R, jnp.ndarray), f"{system_name}: R is not JAX array"
        
        # Design LQR
        K, _, _, success = design_lqr(system.A, system.B, custom_Q=Q, custom_R=R)
        
        if not success:
            return False, f"LQR design failed"
        
        assert isinstance(K, jnp.ndarray), f"{system_name}: K is not JAX array"
        
        # Sample IC
        x0 = system.sample_initial_condition()
        assert isinstance(x0, np.ndarray), f"{system_name}: x0 should be numpy array"
        
        # Quick simulation
        t, X, U = simulate_lqr_controlled_system(
            system.A, system.B, K, x0,
            t_span=1.0, dt=0.02, process_noise_std=0.01,
            early_stop=True
        )
        
        # Check outputs
        assert len(t) > 0, f"{system_name}: No timesteps generated"
        assert X.shape[0] == len(t), f"{system_name}: X shape mismatch"
        assert U.shape[0] == len(t), f"{system_name}: U shape mismatch"
        assert X.shape[1] == system.n_states, f"{system_name}: X dimension mismatch"
        assert U.shape[1] == system.n_inputs, f"{system_name}: U dimension mismatch"
        
        return True, "OK"
        
    except Exception as e:
        return False, str(e)


def main():
    print("="*80)
    print("JAX SYSTEMS TEST SUITE")
    print("="*80)
    
    # Check JAX backend
    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Get systems to test
    systems_to_test = [s for s in ALL_SYSTEMS if s != 'LTISystem']
    
    print(f"\nTesting {len(systems_to_test)} systems...")
    print("="*80)
    
    results = []
    
    for i, system_name in enumerate(systems_to_test, 1):
        print(f"\n[{i}/{len(systems_to_test)}] Testing {system_name}...", end=" ")
        success, message = test_system(system_name)
        
        if success:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
        
        results.append((system_name, success, message))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"\nTotal: {len(results)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    
    if failed > 0:
        print("\nFailed systems:")
        for name, success, message in results:
            if not success:
                print(f"  - {name}: {message}")
    
    print("\n" + "="*80)
    
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())

