"""
Quick test script to verify the setup works correctly
"""

import numpy as np
import sys

print("="*70)
print(" TESTING UNIVERSAL LQR TRANSFORMER SETUP ".center(70, "="))
print("="*70)

# Test 1: Import config
print("\n[1/7] Testing config import...")
try:
    import config
    print("  [OK] Config imported successfully")
    print(f"    - Sequence length: {config.SEQUENCE_LENGTH}")
    print(f"    - Time horizon: {config.TIME_HORIZON}s")
    print(f"    - Sampling rate: {config.DT}s")
except Exception as e:
    print(f"  [X] Failed: {e}")
    sys.exit(1)

# Test 2: Import LQR controller
print("\n[2/7] Testing LQR controller...")
try:
    from lqr_controller import design_lqr, verify_stability
    
    # Test simple system
    A = np.array([[0, 1], [-2, -0.5]])
    B = np.array([[0], [1]])
    K, Q, R, success = design_lqr(A, B, Q_weight=1.0, R_weight=0.1)
    
    if success:
        is_stable, max_eig, _ = verify_stability(A, B, K)
        print(f"  [OK] LQR controller working")
        print(f"    - Stable: {is_stable}")
        print(f"    - Max eigenvalue real part: {max_eig:.6f}")
    else:
        print("  [X] LQR design failed")
        sys.exit(1)
except Exception as e:
    print(f"  [X] Failed: {e}")
    sys.exit(1)

# Test 3: Import all systems
print("\n[3/7] Testing system imports...")
try:
    from systems import *
    from data_generation import get_all_system_classes
    
    system_classes = get_all_system_classes()
    print(f"  [OK] All systems imported successfully")
    print(f"    - Total system families: {len(system_classes)}")
except Exception as e:
    print(f"  [X] Failed: {e}")
    sys.exit(1)

# Test 4: Test a few systems
print("\n[4/7] Testing system instantiation...")
try:
    test_systems = [CartPole(), InvertedPendulum(), QuadrotorHover(), DCMotor()]
    
    for system in test_systems:
        # Check matrices
        assert system.A.shape[0] == system.n_states
        assert system.B.shape[0] == system.n_states
        
        # Test LQR design
        Q, R = system.get_default_lqr_weights()
        K, _, _, success = design_lqr(system.A, system.B, custom_Q=Q, custom_R=R)
        
        if success:
            print(f"  [OK] {system.name}: n_states={system.n_states}, n_inputs={system.n_inputs}, LQR stable")
        else:
            print(f"  [X] {system.name}: LQR failed")
    
except Exception as e:
    print(f"  [X] Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test transformer model
print("\n[5/7] Testing transformer model...")
try:
    import torch
    from transformer_model import UniversalLQRTransformer, count_parameters
    
    model = UniversalLQRTransformer(
        max_state_dim=12,
        max_control_dim=4,
        d_model=128,  # Smaller for testing
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        max_seq_len=64
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    states = torch.randn(batch_size, seq_len, 12)
    controls = model(states)
    
    assert controls.shape == (batch_size, seq_len, 4)
    
    print(f"  [OK] Transformer model working")
    print(f"    - Parameters: {count_parameters(model):,}")
    print(f"    - Input shape: {states.shape}")
    print(f"    - Output shape: {controls.shape}")
except Exception as e:
    print(f"  [X] Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test data generation (single trajectory)
print("\n[6/7] Testing data generation...")
try:
    from data_generation import generate_trajectory
    
    system = CartPole()
    Q, R = system.get_default_lqr_weights()
    K, _, _, success = design_lqr(system.A, system.B, custom_Q=Q, custom_R=R)
    
    t, X, U = generate_trajectory(system, K, trajectory_idx=0)
    
    print(f"  [OK] Data generation working")
    print(f"    - Trajectory length: {len(t)} timesteps")
    print(f"    - State shape: {X.shape}")
    print(f"    - Control shape: {U.shape}")
except Exception as e:
    print(f"  [X] Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test training components
print("\n[7/7] Testing training components...")
try:
    from train import LQRDataset
    
    # Create dummy data
    dummy_data = [{
        'system_name': 'Test',
        'variant_idx': 0,
        'n_states': 4,
        'n_inputs': 1,
        'trajectories': [
            {
                'time': np.arange(0, 10, 0.02),
                'states': np.random.randn(500, 4),
                'controls': np.random.randn(500, 1)
            }
        ],
        'lqr_gain': np.random.randn(1, 4),
        'system_params': {}
    }]
    
    dataset = LQRDataset(dummy_data, sequence_length=32, normalization='standardize')
    
    sample = dataset[0]
    
    print(f"  [OK] Training components working")
    print(f"    - Dataset size: {len(dataset)}")
    print(f"    - Sample keys: {list(sample.keys())}")
except Exception as e:
    print(f"  [X] Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print(" ALL TESTS PASSED! ".center(70, "="))
print("="*70)
print("\nSetup is working correctly. You can now:")
print("  1. Generate data: python data_generation.py")
print("  2. Train model: python train.py")
print("  3. Or run both: python main.py --mode all")
print()

