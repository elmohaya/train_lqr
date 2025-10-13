#!/usr/bin/env python3
"""
Fix in-place assignments in JAX system files
Convert array[idx] = value to array = array.at[idx].set(value)
"""

import os
import re

def fix_inplace_assignments(filepath):
    """Fix in-place assignments in a file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        original_line = line
        
        # Pattern: variable[slice] = expression
        # Match: A[0:2, 2:4] = jnp.eye(2)
        # Convert to: A = A.at[0:2, 2:4].set(jnp.eye(2))
        
        match = re.match(r'^(\s*)([A-Za-z_]\w*)\[([^\]]+)\]\s*=\s*(.+)$', line)
        
        if match and 'def ' not in line and '.at[' not in line:
            indent = match.group(1)
            var_name = match.group(2)
            index = match.group(3)
            value = match.group(4).rstrip()
            
            # Skip if it's a dictionary or self attribute
            if var_name not in ['self', 'params', 'kwargs']:
                # Convert to JAX .at[] syntax
                new_line = f"{indent}{var_name} = {var_name}.at[{index}].set({value})\n"
                line = new_line
                modified = True
        
        new_lines.append(line)
    
    if modified:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
    
    return modified

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    systems_dir = os.path.join(script_dir, 'systems')
    
    system_files = [
        'robotics_systems.py',
        'aerospace_systems.py',
    ]
    
    print("="*80)
    print("FIXING IN-PLACE ASSIGNMENTS FOR JAX")
    print("="*80)
    
    for filename in system_files:
        filepath = os.path.join(systems_dir, filename)
        
        print(f"\nProcessing {filename}...")
        was_fixed = fix_inplace_assignments(filepath)
        
        if was_fixed:
            print(f"  ✓ Fixed in-place assignments")
        else:
            print(f"  - No changes needed")
    
    print("\n" + "="*80)
    print("DONE! Now test with: python test_jax_systems.py")
    print("="*80)

if __name__ == '__main__':
    main()

