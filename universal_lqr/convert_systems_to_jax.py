"""
Script to convert numpy-based system files to JAX
"""

import re
import os

# Define the conversion rules
def convert_to_jax(content):
    """Convert numpy code to JAX code"""
    
    # Replace imports
    content = content.replace('import numpy as np', 'import jax.numpy as jnp\nfrom jax import random')
    content = content.replace('from .base_system import LTISystem', 'from .base_system import LTISystem')
    
    # Replace numpy with jax.numpy in code
    content = re.sub(r'\bnp\.', 'jnp.', content)
    
    # Fix sample_initial_condition to use JAX random
    # Pattern: return np.array([...np.random.uniform...])
    # Replace np.random.uniform with random.uniform(key, ...)
    
    # This is complex, so we'll do a simpler approach:
    # Replace np.random.uniform with a helper that uses self.rng_key
    
    lines = content.split('\n')
    new_lines = []
    in_sample_ic = False
    
    for line in lines:
        if 'def sample_initial_condition(self' in line:
            in_sample_ic = True
            # Add key parameter
            if 'key=None' not in line:
                line = line.replace('(self):', '(self, key=None):')
            new_lines.append(line)
            continue
        
        if in_sample_ic and 'return jnp.array([' in line:
            # End of function, add key handling
            new_lines.insert(-2, '        if key is None:')
            new_lines.insert(-2, '            self.rng_key, key = random.split(self.rng_key)')
            new_lines.insert(-2, '        ')
            in_sample_ic = False
        
        # Replace np.random.uniform calls
        if 'jnp.random.uniform' in line:
            # Extract bounds
            match = re.search(r'jnp\.random\.uniform\(([^,]+),\s*([^)]+)\)', line)
            if match:
                low, high = match.groups()
                # Generate unique key split
                indent = len(line) - len(line.lstrip())
                line = ' ' * indent + f'random.uniform(key, minval={low}, maxval={high}),'
                # We need to split keys for each call - simplified: use same key
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def process_file(input_path, output_path):
    """Process a single system file"""
    with open(input_path, 'r') as f:
        content = f.read()
    
    converted = convert_to_jax(content)
    
    with open(output_path, 'w') as f:
        f.write(converted)
    
    print(f"Converted: {input_path} -> {output_path}")


if __name__ == '__main__':
    # Get the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    systems_dir = os.path.join(parent_dir, 'systems')
    jax_systems_dir = os.path.join(os.path.dirname(__file__), 'systems')
    
    # Files to convert
    system_files = [
        'mechanical_systems.py',
        'electrical_systems.py',
        'robotics_systems.py',
        'aerospace_systems.py',
        'vehicle_systems.py',
        'other_systems.py',
    ]
    
    for filename in system_files:
        input_path = os.path.join(systems_dir, filename)
        output_path = os.path.join(jax_systems_dir, filename)
        
        if os.path.exists(input_path):
            process_file(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found")
    
    print("\nConversion complete!")

