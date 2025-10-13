"""
Aerospace LTI Systems
"""

import jax.numpy as jnp
import numpy as np
from .base_system import LTISystem


class QuadrotorHover(LTISystem):
    """
    Quadrotor UAV (linearized at hover)
    State: [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]
    Input: [thrust_deviation, tau_phi, tau_theta, tau_psi]
    """
    
    def get_default_params(self):
        return {
            'm': 0.8,      # mass (kg) - increased
            'g': 9.81,     # gravity
            'Ix': 0.03,    # moment of inertia x - increased
            'Iy': 0.03,    # moment of inertia y - increased
            'Iz': 0.05,    # moment of inertia z - increased
            'l': 0.2,      # arm length (m) - reduced
            'b': 1e-6,     # thrust coefficient
            'd': 1e-7      # drag coefficient
        }
    
    def get_matrices(self):
        m, g = self.params['m'], self.params['g']
        Ix, Iy, Iz = self.params['Ix'], self.params['Iy'], self.params['Iz']
        
        # 12-state model with damping
        A = jnp.zeros((12, 12))
        # Position derivatives
        A = A.at[0:3, 6:9].set(jnp.eye(3))
        # Velocity from angles (small angle approximation) with damping
        A = A.at[6, 4].set(g)   # x_ddot from theta
        A = A.at[6, 6].set(-0.5)  # air drag
        A = A.at[7, 3].set(-g)  # y_ddot from phi
        A = A.at[7, 7].set(-0.5)  # air drag
        A = A.at[8, 8].set(-0.5)  # air drag
        # Attitude derivatives
        A = A.at[3:6, 9:12].set(jnp.eye(3))
        # Angular damping
        A = A.at[9, 9].set(-0.1)
        A = A.at[10, 10].set(-0.1)
        A = A.at[11, 11].set(-0.1)
        
        B = jnp.zeros((12, 4))
        B = B.at[8, 0].set(1/m)  # z acceleration from thrust
        B = B.at[9, 1].set(1/Ix)  # roll acceleration from tau_phi
        B = B.at[10, 2].set(1/Iy)  # pitch acceleration from tau_theta
        B = B.at[11, 3].set(1/Iz)  # yaw acceleration from tau_psi
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([20.0, 20.0, 50.0,  # position - higher
                     100.0, 100.0, 20.0,  # attitude - much higher
                     2.0, 2.0, 5.0,       # linear velocity
                     10.0, 10.0, 2.0]))    # angular velocity
        R = jnp.diag(jnp.array([10.0, 10.0, 10.0, 10.0]))  # Much higher R
        return Q, R
    
    def sample_initial_condition(self):
        # Quadrotor: flying in 6m³ space, tilt ±20°, realistic flight speeds
        return np.array([
            np.random.uniform(-3.0, 3.0),   # x (m)
            np.random.uniform(-3.0, 3.0),   # y (m)
            np.random.uniform(-2.0, 2.0),   # z (m)
            np.random.uniform(-0.35, 0.35), # phi (rad) ≈ ±20° roll
            np.random.uniform(-0.35, 0.35), # theta (rad) ≈ ±20° pitch
            np.random.uniform(-0.5, 0.5),   # psi (rad) ≈ ±30° yaw
            np.random.uniform(-2.0, 2.0),   # x_dot (m/s)
            np.random.uniform(-2.0, 2.0),   # y_dot (m/s)
            np.random.uniform(-1.5, 1.5),   # z_dot (m/s)
            np.random.uniform(-1.0, 1.0),   # p (rad/s) - roll rate
            np.random.uniform(-1.0, 1.0),   # q (rad/s) - pitch rate
            np.random.uniform(-1.0, 1.0)    # r (rad/s) - yaw rate
        ])


class FixedWingAircraft(LTISystem):
    """
    Fixed-wing aircraft longitudinal dynamics (short period mode)
    State: [u, w, q, theta] (forward vel, vertical vel, pitch rate, pitch angle)
    Input: [elevator_deflection]
    """
    
    def get_default_params(self):
        return {
            'Xu': -0.2,    # stability derivatives - more damping
            'Xw': 0.05,    # reduced
            'Zu': -0.5,    # increased
            'Zw': -3.0,    # more damping
            'Zq': -8.0,    # increased
            'Mu': 0.0,
            'Mw': -1.0,    # more damping
            'Mq': -5.0,    # much more damping
            'g': 9.81,
            'U0': 30.0     # trim velocity (m/s) - reduced
        }
    
    def get_matrices(self):
        p = self.params
        g, U0 = p['g'], p['U0']
        
        A = jnp.array([
            [p['Xu'], p['Xw'], 0, -g],
            [p['Zu'], p['Zw'], U0+p['Zq'], 0],
            [p['Mu'], p['Mw'], p['Mq'], 0],
            [0, 0, 1, -0.01]  # Added small damping
        ])
        
        # Control derivatives (elevator) - reduced
        B = jnp.array([
            [0.05],    # Xde - reduced
            [-2.0],    # Zde - reduced
            [-5.0],    # Mde - reduced
            [0]
        ])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([10.0, 20.0, 50.0, 100.0]))  # Much higher penalties
        R = jnp.array([[100.0]])  # Much higher R
        return Q, R
    
    def sample_initial_condition(self):
        # Fixed-wing: speed perturbations ±10 m/s, altitude ±5m, pitch ±30°
        return np.array([
            np.random.uniform(-10.0, 10.0),  # u - forward speed perturbation (m/s)
            np.random.uniform(-5.0, 5.0),    # w - vertical speed perturbation (m/s)
            np.random.uniform(-0.5, 0.5),    # q - pitch rate (rad/s)
            np.random.uniform(-0.5, 0.5)     # theta - pitch angle (rad) ≈ ±30°
        ])


class VTOLLinearized(LTISystem):
    """
    VTOL aircraft (vertical takeoff and landing, linearized)
    State: [x, z, theta, x_dot, z_dot, theta_dot]
    Input: [thrust, moment]
    """
    
    def get_default_params(self):
        return {
            'm': 3.0,     # mass (kg) - reduced
            'J': 0.05,    # moment of inertia - increased
            'g': 9.81,
            'l': 0.2      # moment arm - reduced
        }
    
    def get_matrices(self):
        m, J, g, l = self.params['m'], self.params['J'], self.params['g'], self.params['l']
        
        # Linearized around hover (theta = 0) with damping
        A = jnp.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, -g, -0.5, 0, 0],  # Added air drag
            [0, 0, 0, 0, -0.5, 0],   # Added air drag
            [0, 0, 0, 0, 0, -0.2]    # Added angular damping
        ])
        B = jnp.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1/m, 0],
            [0, 1/J]
        ])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([50.0, 100.0, 100.0, 5.0, 10.0, 10.0]))  # Much higher penalties
        R = jnp.diag(jnp.array([10.0, 10.0]))  # Much higher R
        return Q, R
    
    def sample_initial_condition(self):
        # VTOL: position ±3m, tilt ±30°, realistic velocities
        return np.array([
            np.random.uniform(-3.0, 3.0),   # x (m)
            np.random.uniform(-3.0, 3.0),   # z (m)
            np.random.uniform(-0.5, 0.5),   # theta (rad) ≈ ±30° pitch
            np.random.uniform(-2.0, 2.0),   # x_dot (m/s)
            np.random.uniform(-2.0, 2.0),   # z_dot (m/s)
            np.random.uniform(-1.0, 1.0)    # theta_dot (rad/s)
        ])

