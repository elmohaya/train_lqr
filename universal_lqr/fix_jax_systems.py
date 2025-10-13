#!/usr/bin/env python3
"""
Fix JAX system files to be properly JAX-compatible
- Convert np.diag([...]) to jnp.diag(jnp.array([...]))
- Fix in-place assignments
"""

import os
import re

def fix_jax_file(filepath):
    """Fix a single system file for JAX compatibility"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix 1: Convert np.diag([list]) to jnp.diag(jnp.array([list]))
    # Pattern: jnp.diag([...])
    def fix_diag(match):
        list_content = match.group(1)
        return f'jnp.diag(jnp.array([{list_content}]))'
    
    content = re.sub(r'jnp\.diag\(\[([^\]]+)\]\)', fix_diag, content)
    
    # Fix 2: Convert in-place array assignments
    # Pattern: array[index] = value
    # This is trickier - need to check context
    
    # Fix 3: Ensure random.uniform is used correctly with keys
    # Already using jnp.random.uniform which should work
    
    # Fix 4: Make sure all matrices use jnp.array() properly
    # Check for patterns like: A = jnp.array([ without proper nesting
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def find_inplace_assignments(filepath):
    """Find in-place assignments that need fixing"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    issues = []
    for i, line in enumerate(lines, 1):
        # Look for array[...] = ... patterns (excluding comments)
        if '=' in line and '[' in line and ']' in line:
            # Skip if it's already using .at[].set()
            if '.at[' not in line and 'def ' not in line:
                # Check if it looks like in-place assignment
                match = re.search(r'(\w+)\[([^\]]+)\]\s*=\s*([^#\n]+)', line)
                if match:
                    var_name = match.group(1)
                    index = match.group(2)
                    value = match.group(3).strip()
                    # Skip if it's a list or dict
                    if var_name not in ['self', 'params', 'return']:
                        issues.append({
                            'line': i,
                            'content': line.strip(),
                            'var': var_name,
                            'index': index,
                            'value': value
                        })
    return issues

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
    print("FIXING JAX SYSTEM FILES")
    print("="*80)
    
    total_fixed = 0
    total_issues = 0
    
    for filename in system_files:
        filepath = os.path.join(systems_dir, filename)
        
        print(f"\nProcessing {filename}...")
        
        # Fix diag issues
        was_fixed = fix_jax_file(filepath)
        if was_fixed:
            print(f"  ✓ Fixed jnp.diag() calls")
            total_fixed += 1
        
        # Find in-place assignment issues
        issues = find_inplace_assignments(filepath)
        if issues:
            print(f"  ⚠ Found {len(issues)} potential in-place assignments:")
            for issue in issues[:3]:  # Show first 3
                print(f"    Line {issue['line']}: {issue['content'][:60]}...")
            total_issues += len(issues)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Files with diag() fixes: {total_fixed}")
    print(f"  In-place assignments found: {total_issues}")
    
    if total_issues > 0:
        print("\n  Note: In-place assignments need manual fixing.")
        print("  JAX arrays are immutable - use: array = array.at[idx].set(value)")
    
    print("\n  Re-run: python test_jax_systems.py")
    print("="*80)

if __name__ == '__main__':
    main()

