#!/usr/bin/env python3
"""
Simple script to convert numpy-based systems to JAX
Only replaces np. with jnp. - keeps everything else the same
"""

import os
import re

def convert_file(input_path, output_path):
    """Convert a numpy system file to JAX"""
    with open(input_path, 'r') as f:
        content = f.read()
    
    # Simple replacements
    content = content.replace('import numpy as np', 'import jax.numpy as jnp')
    content = re.sub(r'\bnp\.', 'jnp.', content)
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Converted: {os.path.basename(input_path)}")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    systems_dir = os.path.join(parent_dir, 'systems')
    jax_systems_dir = os.path.join(script_dir, 'systems')
    
    # Files to convert
    system_files = [
        'mechanical_systems.py',
        'electrical_systems.py',
        'robotics_systems.py',
        'aerospace_systems.py',
        'vehicle_systems.py',
        'other_systems.py',
    ]
    
    print("Converting system files to JAX...")
    print("="*50)
    
    for filename in system_files:
        input_path = os.path.join(systems_dir, filename)
        output_path = os.path.join(jax_systems_dir, filename)
        
        if os.path.exists(input_path):
            convert_file(input_path, output_path)
        else:
            print(f"✗ Not found: {filename}")
    
    # Also create __init__.py
    init_content = """\"\"\"
LTI System Families for Universal LQR Transformer (JAX version)
\"\"\"

from .base_system import LTISystem
from .mechanical_systems import *
from .electrical_systems import *
from .robotics_systems import *
from .vehicle_systems import *
from .aerospace_systems import *
from .other_systems import *

# Import all system classes for easy access
__all__ = [
    'LTISystem',
    # Mechanical (13)
    'MassSpringDamper',
    'SimplePendulum',
    'InvertedPendulum',
    'DoublePendulum',
    'CartPole',
    'Acrobot',
    'FurutaPendulum',
    'BallAndBeam',
    'BallAndPlate',
    'ReactionWheelPendulum',
    'FlexibleBeam',
    'MagneticLevitation',
    'SuspensionSystem',
    # Electrical (1)
    'DCMotor',
    # Robotics (12)
    'TwoLinkArm',
    'ThreeLinkManipulator',
    'UnicycleRobot',
    'DifferentialDriveRobot',
    'SCARARobot',
    'SegwayRobot',
    'OmnidirectionalRobot',
    'CableDrivenRobot',
    'FlexibleJointRobot',
    'PlanarBiped',
    'SixDOFManipulator',
    'DualArmRobot',
    # Vehicles (3)
    'VehicleLateralDynamics',
    'LongitudinalCruiseControl',
    'PlatooningModel',
    # Aerospace (3)
    'QuadrotorHover',
    'FixedWingAircraft',
    'VTOLLinearized',
    # Other (3)
    'DoubleIntegrator',
    'LotkaVolterra',
    'ChemicalReactor',
]
"""
    
    with open(os.path.join(jax_systems_dir, '__init__.py'), 'w') as f:
        f.write(init_content)
    
    print("✓ Created __init__.py")
    print("="*50)
    print("✓ Conversion complete!")

if __name__ == '__main__':
    main()

