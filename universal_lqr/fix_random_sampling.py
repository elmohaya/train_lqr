#!/usr/bin/env python3
"""
Fix random sampling in JAX system files
JAX random functions use jnp.random (old style) which still works
But we should use numpy for IC sampling (not performance critical)
"""

import os
import re

def fix_random_sampling(filepath):
    """Fix random sampling to use numpy instead of jax.numpy"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # In sample_initial_condition methods, use numpy random
    # Change jnp.random.uniform to np.random.uniform
    # But we need to import numpy as np
    
    # Check if numpy is imported
    if 'import numpy as np' not in content:
        # Add numpy import after jax.numpy import
        content = content.replace(
            'import jax.numpy as jnp',
            'import jax.numpy as jnp\nimport numpy as np'
        )
    
    # In sample_initial_condition methods, replace jnp with np for random and array creation
    # Pattern: find sample_initial_condition method and replace within it
    
    def replace_in_method(match):
        method_content = match.group(0)
        # Replace jnp.random with np.random in this method
        method_content = method_content.replace('jnp.random.', 'np.random.')
        # Also replace jnp.array with np.array for return values
        method_content = re.sub(r'return jnp\.array\(', 'return np.array(', method_content)
        return method_content
    
    # Match sample_initial_condition method
    pattern = r'def sample_initial_condition\(self\):.*?(?=\n    def |\nclass |\Z)'
    content = re.sub(pattern, replace_in_method, content, flags=re.DOTALL)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    systems_dir = os.path.join(script_dir, 'systems')
    
    system_files = [
        'mechanical_systems.py',
        'electrical_systems.py',
        'robotics_systems.py',
        'aerospace_systems.py',
        'vehicle_systems.py',
        'other_systems.py',
    ]
    
    print("="*80)
    print("FIXING RANDOM SAMPLING FOR JAX")
    print("="*80)
    print("\nNote: IC sampling uses numpy (not performance-critical)")
    
    for filename in system_files:
        filepath = os.path.join(systems_dir, filename)
        
        print(f"\nProcessing {filename}...")
        was_fixed = fix_random_sampling(filepath)
        
        if was_fixed:
            print(f"  ✓ Fixed random sampling")
        else:
            print(f"  - No changes needed")
    
    print("\n" + "="*80)
    print("DONE! Test with: python test_jax_systems.py")
    print("="*80)

if __name__ == '__main__':
    main()

