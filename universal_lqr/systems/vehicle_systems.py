"""
Vehicle LTI Systems
"""

import jax.numpy as jnp
import numpy as np
from .base_system import LTISystem


class VehicleLateralDynamics(LTISystem):
    """
    Vehicle lateral dynamics (bicycle model)
    State: [lateral_velocity, yaw_rate, lateral_position, yaw_angle]
    Input: [steering_angle]
    """
    
    def get_default_params(self):
        return {
            'm': 1200.0,   # vehicle mass (kg) - reduced
            'Iz': 2000.0,  # yaw inertia (kg·m^2) - reduced
            'lf': 1.0,     # distance to front axle (m) - reduced
            'lr': 1.4,     # distance to rear axle (m) - reduced
            'Cf': 15000,   # front cornering stiffness (N/rad) - reduced
            'Cr': 25000,   # rear cornering stiffness (N/rad) - reduced
            'Vx': 15.0     # longitudinal velocity (m/s) - reduced
        }
    
    def get_matrices(self):
        m, Iz, lf, lr, Cf, Cr, Vx = (self.params['m'], self.params['Iz'], 
                                      self.params['lf'], self.params['lr'],
                                      self.params['Cf'], self.params['Cr'], 
                                      self.params['Vx'])
        
        A = jnp.array([
            [-(Cf+Cr)/(m*Vx), -(Cf*lf-Cr*lr)/(m*Vx)-Vx, 0, 0],
            [-(Cf*lf-Cr*lr)/(Iz*Vx), -(Cf*lf**2+Cr*lr**2)/(Iz*Vx), 0, 0],
            [1, 0, -0.01, Vx],  # Added small damping
            [0, 1, 0, -0.01]     # Added small damping
        ])
        B = jnp.array([[Cf/m], [Cf*lf/Iz], [0], [0]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([10.0, 20.0, 100.0, 50.0]))  # Much higher penalties on position/heading
        R = jnp.array([[100.0]])  # Much higher R
        return Q, R
    
    def sample_initial_condition(self):
        # Vehicle lateral: lane deviation ±3m, heading ±30°, realistic lateral dynamics
        return np.array([
            np.random.uniform(-3.0, 3.0),   # lateral velocity (m/s)
            np.random.uniform(-1.0, 1.0),   # yaw rate (rad/s)
            np.random.uniform(-3.0, 3.0),   # lateral position (m) - lane width
            np.random.uniform(-0.5, 0.5)    # yaw angle (rad) ≈ ±30°
        ])


class LongitudinalCruiseControl(LTISystem):
    """
    Longitudinal vehicle dynamics (cruise control)
    State: [position, velocity]
    Input: [throttle/brake]
    """
    
    def get_default_params(self):
        return {
            'm': 1200.0,   # mass (kg) - reduced
            'Cd': 0.25,    # drag coefficient - reduced
            'A': 2.0,      # frontal area (m^2) - reduced
            'rho': 1.225,  # air density (kg/m^3)
            'Cr': 0.015,   # rolling resistance - increased
            'g': 9.81
        }
    
    def get_matrices(self):
        m = self.params['m']
        Cd, A, rho = self.params['Cd'], self.params['A'], self.params['rho']
        Cr, g = self.params['Cr'], self.params['g']
        
        # Linearized around nominal velocity v0 = 15 m/s (reduced)
        v0 = 15.0
        drag_coeff = 0.5 * Cd * A * rho
        
        A_mat = jnp.array([
            [0, 1],
            [0, -2*drag_coeff*v0/m - Cr*g - 0.5]  # Added extra damping
        ])
        B_mat = jnp.array([[0], [1/m]])
        return A_mat, B_mat
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([10.0, 100.0]))  # Much higher penalties
        R = jnp.array([[10.0]])  # Higher R
        return Q, R
    
    def sample_initial_condition(self):
        # Cruise control: position deviation ±50m, speed deviation ±10 m/s
        return np.array([
            np.random.uniform(-50.0, 50.0),  # position deviation (m)
            np.random.uniform(-10.0, 10.0)   # velocity deviation (m/s) from cruise speed
        ])


class PlatooningModel(LTISystem):
    """
    Vehicle platooning (string of 3 vehicles, predecessor following)
    State: [p1, v1, p2, v2, p3, v3] (positions and velocities)
    Input: [a1, a2, a3] (accelerations)
    """
    
    def get_default_params(self):
        return {
            'tau': 0.5,    # time constant (s)
            'k_p': 0.5,    # position feedback gain
            'k_v': 1.0     # velocity feedback gain
        }
    
    def get_matrices(self):
        tau = self.params['tau']
        k_p = self.params['k_p']
        k_v = self.params['k_v']
        
        # Each vehicle follows the one ahead
        A = jnp.array([
            [0, 1, 0, 0, 0, 0],
            [0, -1/tau, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [k_p/tau, k_v/tau, 0, -1/tau, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, k_p/tau, k_v/tau, 0, -1/tau]
        ])
        B = jnp.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1]
        ])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([10.0, 1.0, 10.0, 1.0, 10.0, 1.0]))
        R = jnp.diag(jnp.array([1.0, 1.0, 1.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # Platooning: distance gaps ±15m, speed differences ±5 m/s per vehicle
        return np.array([
            np.random.uniform(-15.0, 15.0),  # position 1 (m)
            np.random.uniform(-5.0, 5.0),    # velocity 1 (m/s)
            np.random.uniform(-15.0, 15.0),  # position 2 (m)
            np.random.uniform(-5.0, 5.0),    # velocity 2 (m/s)
            np.random.uniform(-15.0, 15.0),  # position 3 (m)
            np.random.uniform(-5.0, 5.0)     # velocity 3 (m/s)
        ])

