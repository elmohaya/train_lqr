"""
Script to fix R matrices across all LTI systems.

The problem: R matrices (control cost) are too small, leading to huge control signals.
Solution: Increase R by 10-100x to properly penalize control effort.

Strategy:
- For unstable systems (pendulums, etc.): R = 10-50
- For stable systems with large Q: R = 10-100  
- For multi-input systems: R = 10-50 per input
"""

import re
from pathlib import Path

# Define new R scaling factors for each system
# Key: system class name, Value: R scaling factor (multiply current R by this)
R_SCALING = {
    # === CRITICAL FIXES (systems with huge control signals) ===
    'InvertedPendulum': 10,  # R: 0.01 -> 10
    'SegwayRobot': 10,        # R: 0.1 -> 10
    'FurutaPendulum': 10,       # R: 1.0 -> 50
    'ReactionWheelPendulum': 10, # R: 1.0 -> 50
    'SCARARobot': 10,           # R: 1.0 -> 50
    'SixDOFManipulator': 10,    # R: 1.0 -> 50
    'QuadrotorHover': 10,       # R: 1.0 -> 50
    'FlexibleJointRobot': 10,   # R: likely 1.0 -> 50
    'FlexibleBeam': 10,         # R: likely 1.0 -> 50
    'SuspensionSystem': 10,     # R: likely 1.0 -> 20
    'PlanarBiped': 10,          # R: 1.0 -> 50
    'SegwayRobot': 10,         # R: 0.1 -> 10
    
    # === OTHER UNSTABLE SYSTEMS (pendulums, balance) ===
    'SimplePendulum': 10,
    'DoublePendulum': 10,
    'Acrobot': 10,
    'BallAndBeam': 10,
    'BallAndPlate': 10,
    'CartPole': 10,
    
    # === COMPLEX MULTI-INPUT SYSTEMS ===
    'ThreeLinkManipulator': 10,
    'DualArmRobot': 10,
    'CableDrivenRobot': 10,
    'OmnidirectionalRobot': 10,
    'FixedWingAircraft': 10,
    'VTOLLinearized': 10,
    
    # === MODERATE ADJUSTMENTS (stable but could be better) ===
    'MassSpringDamper': 10,
    'TwoLinkArm': 10,
    'UnicycleRobot': 10,
    'DifferentialDriveRobot': 10,
    'VehicleLateralDynamics': 10,
    'LongitudinalCruiseControl': 10,
    'PlatooningModel': 10,
    'DCMotor': 10,
    'ACMotor': 10,
    'MagneticLevitation': 10,
    'ChemicalReactor': 10,
    'LotkaVolterra': 10,
    'DoubleIntegrator': 10,
}

def fix_r_matrix_in_file(file_path, system_name, scaling_factor):
    """
    Fix R matrix in a system file by scaling it up.
    
    Args:
        file_path: Path to the system file
        system_name: Name of the system class
        scaling_factor: Factor to multiply R by
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the get_default_lqr_weights method for this system
    # Pattern: class SystemName...def get_default_lqr_weights...R = ...
    
    # Look for R matrix definitions
    # Common patterns:
    # 1. R = 0.01 * jnp.eye(...)
    # 2. R = jnp.array([...])
    # 3. R = jnp.diag(jnp.array([...]))
    
    modified = False
    lines = content.split('\n')
    new_lines = []
    
    in_target_class = False
    in_lqr_method = False
    indent_level = 0
    
    for i, line in enumerate(lines):
        # Check if we're in the target class
        if f'class {system_name}(' in line:
            in_target_class = True
            indent_level = len(line) - len(line.lstrip())
        
        # Check if we're in another class (exit target class)
        elif line.strip().startswith('class ') and in_target_class:
            in_target_class = False
            in_lqr_method = False
        
        # Check if we're in the LQR method
        if in_target_class and 'def get_default_lqr_weights' in line:
            in_lqr_method = True
        
        # Check if we exit the method
        if in_lqr_method and line.strip() and not line.strip().startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and 'def ' in line:
                in_lqr_method = False
        
        # Modify R matrix if we're in the right place
        if in_lqr_method and 'R = ' in line and '=' in line:
            stripped = line.strip()
            
            # Skip if it's a comment
            if stripped.startswith('#'):
                new_lines.append(line)
                continue
            
            # Get indentation
            indent = line[:len(line) - len(line.lstrip())]
            
            # Pattern 1: R = X * jnp.eye(...)
            if 'jnp.eye' in line:
                match = re.search(r'R\s*=\s*([\d.]+)\s*\*\s*jnp\.eye', line)
                if match:
                    old_value = float(match.group(1))
                    new_value = old_value * scaling_factor
                    new_line = re.sub(
                        r'(R\s*=\s*)([\d.]+)(\s*\*\s*jnp\.eye)',
                        f'\\g<1>{new_value}\\g<3>',
                        line
                    )
                    new_lines.append(new_line)
                    modified = True
                    print(f"  {system_name}: R scaled {old_value} -> {new_value}")
                    continue
            
            # Pattern 2: R = jnp.diag(jnp.array([...]))
            elif 'jnp.diag' in line and 'jnp.array' in line:
                # Extract the array values
                match = re.search(r'jnp\.array\(\[([\d.,\s]+)\]\)', line)
                if match:
                    values_str = match.group(1)
                    values = [float(x.strip()) for x in values_str.split(',')]
                    new_values = [v * scaling_factor for v in values]
                    new_values_str = ', '.join([f'{v}' for v in new_values])
                    new_line = line.replace(
                        f'jnp.array([{values_str}])',
                        f'jnp.array([{new_values_str}])'
                    )
                    new_lines.append(new_line)
                    modified = True
                    print(f"  {system_name}: R scaled {values} -> {new_values}")
                    continue
            
            # Pattern 3: R = jnp.array([[X]])
            elif 'jnp.array' in line:
                match = re.search(r'jnp\.array\(\[\[([\d.]+)\]\]\)', line)
                if match:
                    old_value = float(match.group(1))
                    new_value = old_value * scaling_factor
                    new_line = line.replace(
                        f'jnp.array([[{old_value}]])',
                        f'jnp.array([[{new_value}]])'
                    )
                    new_lines.append(new_line)
                    modified = True
                    print(f"  {system_name}: R scaled {old_value} -> {new_value}")
                    continue
        
        new_lines.append(line)
    
    if modified:
        with open(file_path, 'w') as f:
            f.write('\n'.join(new_lines))
        return True
    return False

def main():
    print("="*80)
    print("FIXING R MATRICES ACROSS ALL LTI SYSTEMS")
    print("="*80)
    print()
    
    systems_dir = Path(__file__).parent / 'systems'
    
    # Get all system files
    system_files = [
        systems_dir / 'mechanical_systems.py',
        systems_dir / 'robotics_systems.py',
        systems_dir / 'aerospace_systems.py',
        systems_dir / 'vehicle_systems.py',
        systems_dir / 'electrical_systems.py',
        systems_dir / 'other_systems.py',
    ]
    
    total_fixed = 0
    
    for file_path in system_files:
        if not file_path.exists():
            continue
        
        print(f"Processing {file_path.name}...")
        
        for system_name, scaling_factor in R_SCALING.items():
            # Check if this system is in this file
            with open(file_path, 'r') as f:
                if f'class {system_name}(' in f.read():
                    if fix_r_matrix_in_file(file_path, system_name, scaling_factor):
                        total_fixed += 1
    
    print()
    print("="*80)
    print(f"COMPLETED: Fixed {total_fixed} systems")
    print("="*80)

if __name__ == "__main__":
    main()

