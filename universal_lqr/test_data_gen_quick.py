#!/usr/bin/env python3
"""
Quick test of JAX data generation
"""

import os
import sys

# Modify config temporarily for quick test
import config
original_variants = config.NUM_VARIANTS_PER_SYSTEM
original_trajs = config.NUM_TRAJECTORIES_PER_VARIANT

config.NUM_VARIANTS_PER_SYSTEM = 2  # Just 2 variants
config.NUM_TRAJECTORIES_PER_VARIANT = 10  # Just 10 trajectories

from data_generation_jax import generate_data_jax_accelerated
from systems import __all__ as ALL_SYSTEMS

# Test with just 2 systems
systems_to_test = ['MassSpringDamper', 'CartPole']
output_path = os.path.join(config.DATA_DIR, 'test_jax_quick.h5')

print("="*80)
print("QUICK JAX DATA GENERATION TEST")
print("="*80)
print(f"Testing {len(systems_to_test)} systems")
print(f"Variants: {config.NUM_VARIANTS_PER_SYSTEM}")
print(f"Trajectories per variant: {config.NUM_TRAJECTORIES_PER_VARIANT}")
print("="*80)

try:
    num_sequences = generate_data_jax_accelerated(
        systems_to_test,
        output_path,
        batch_size=10,
        chunk_size=100
    )
    
    print("\n" + "="*80)
    print("✓ TEST PASSED!")
    print(f"Generated {num_sequences:,} sequences successfully")
    print("="*80)
    print("\nJAX data generation is working correctly!")
    print("You can now run the full generation:")
    print("  python data_generation_jax.py")
    
except Exception as e:
    print("\n" + "="*80)
    print("✗ TEST FAILED!")
    print(f"Error: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Restore config
config.NUM_VARIANTS_PER_SYSTEM = original_variants
config.NUM_TRAJECTORIES_PER_VARIANT = original_trajs

