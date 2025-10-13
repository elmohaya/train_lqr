#!/usr/bin/env python3
"""
Better fix for in-place assignments that handles comments correctly
"""

import os
import re

def fix_inplace_assignments_v2(filepath):
    """Fix in-place assignments, preserving comments"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        original_line = line
        
        # Pattern: variable[slice] = expression  # optional comment
        # Extract comment separately
        comment_match = re.search(r'#.*$', line)
        comment = comment_match.group(0) if comment_match else ''
        
        # Remove comment for processing
        line_no_comment = line.split('#')[0] if '#' in line else line
        
        match = re.match(r'^(\s*)([A-Za-z_]\w*)\[([^\]]+)\]\s*=\s*(.+)$', line_no_comment)
        
        if match and 'def ' not in line and '.at[' not in line:
            indent = match.group(1)
            var_name = match.group(2)
            index = match.group(3)
            value = match.group(4).rstrip()
            
            # Skip if it's a dictionary or self attribute
            if var_name not in ['self', 'params', 'kwargs']:
                # Convert to JAX .at[] syntax, preserving comment
                if comment:
                    new_line = f"{indent}{var_name} = {var_name}.at[{index}].set({value})  {comment}\n"
                else:
                    new_line = f"{indent}{var_name} = {var_name}.at[{index}].set({value})\n"
                line = new_line
                modified = True
        else:
            line = original_line
        
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
    print("FIXING IN-PLACE ASSIGNMENTS (V2 - WITH COMMENT HANDLING)")
    print("="*80)
    
    for filename in system_files:
        filepath = os.path.join(systems_dir, filename)
        
        print(f"\nProcessing {filename}...")
        was_fixed = fix_inplace_assignments_v2(filepath)
        
        if was_fixed:
            print(f"  ✓ Fixed in-place assignments")
        else:
            print(f"  - No changes needed")
    
    print("\n" + "="*80)
    print("DONE! Test with: python test_jax_systems.py")
    print("="*80)

if __name__ == '__main__':
    main()

