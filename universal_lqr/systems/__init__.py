"""
LTI System Families for Universal LQR Transformer
"""

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
    # Electrical (1 - kept only DCMotor, removed 7 fast converters/circuits)
    'DCMotor',
    # Robotics (12 - added 8 new diverse systems!)
    'TwoLinkArm',
    'ThreeLinkManipulator',
    'UnicycleRobot',
    'DifferentialDriveRobot',
    'SCARARobot',             # NEW
    'SegwayRobot',            # NEW
    'OmnidirectionalRobot',   # NEW
    # 'PlanarQuadruped',      # Excluded - LQR fails (too complex)
    'CableDrivenRobot',       # NEW
    'FlexibleJointRobot',     # NEW
    'PlanarBiped',            # NEW
    'SixDOFManipulator',      # NEW
    'DualArmRobot',           # NEW
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

