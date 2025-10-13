"""
LTI System Families for Universal LQR Transformer (JAX version)
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
