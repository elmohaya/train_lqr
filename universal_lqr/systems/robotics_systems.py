"""
Robotics LTI Systems
"""

import jax.numpy as jnp
import numpy as np
from .base_system import LTISystem


class TwoLinkArm(LTISystem):
    """
    Two-link planar robotic arm (linearized)
    State: [theta1, theta2, theta1_dot, theta2_dot]
    Input: [torque1, torque2]
    """
    
    def get_default_params(self):
        return {
            'm1': 1.0,   # link 1 mass (kg)
            'm2': 1.0,   # link 2 mass (kg)
            'l1': 0.5,   # link 1 length (m)
            'l2': 0.5,   # link 2 length (m)
            'lc1': 0.25, # link 1 center of mass
            'lc2': 0.25, # link 2 center of mass
            'I1': 0.02,  # link 1 inertia
            'I2': 0.02,  # link 2 inertia
            'b1': 0.1,   # joint 1 damping
            'b2': 0.1,   # joint 2 damping
            'g': 9.81
        }
    
    def get_matrices(self):
        # Linearized around horizontal configuration (theta1=0, theta2=0)
        m1, m2, l1, l2, lc1, lc2 = (self.params['m1'], self.params['m2'], 
                                     self.params['l1'], self.params['l2'],
                                     self.params['lc1'], self.params['lc2'])
        I1, I2, b1, b2, g = (self.params['I1'], self.params['I2'], 
                             self.params['b1'], self.params['b2'], self.params['g'])
        
        # Simplified mass matrix at equilibrium
        M11 = I1 + I2 + m2*l1**2
        M12 = I2 + m2*l1*lc2
        M22 = I2
        
        # Gravity terms (linearized)
        G1 = (m1*lc1 + m2*l1)*g
        G2 = m2*lc2*g
        
        # Inverse mass matrix
        det_M = M11*M22 - M12**2
        Minv = jnp.array([[M22/det_M, -M12/det_M],
                        [-M12/det_M, M11/det_M]])
        
        A = jnp.zeros((4, 4))
        A = A.at[0:2, 2:4].set(jnp.eye(2))
        A = A.at[2:4, 0:2].set(-Minv @ jnp.array([[G1, 0], [0, G2]]))
        A = A.at[2:4, 2:4].set(-Minv @ jnp.array([[b1, 0], [0, b2]]))
        
        B = jnp.zeros((4, 2))
        B = B.at[2:4, :].set(Minv)
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))
        R = jnp.diag(jnp.array([1.0, 1.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # Robot arm: joints can start ±60° with realistic velocities
        return np.array([
            np.random.uniform(-1.0, 1.0),   # theta1 (rad) ≈ ±60°
            np.random.uniform(-1.0, 1.0),   # theta2 (rad) ≈ ±60°
            np.random.uniform(-1.5, 1.5),   # theta1_dot (rad/s)
            np.random.uniform(-1.5, 1.5)    # theta2_dot (rad/s)
        ])


class ThreeLinkManipulator(LTISystem):
    """
    Three-link manipulator (simplified linearized model)
    State: [theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot]
    Input: [torque1, torque2, torque3]
    """
    
    def get_default_params(self):
        return {
            'm1': 1.0, 'm2': 1.0, 'm3': 1.0,
            'l1': 0.4, 'l2': 0.4, 'l3': 0.4,
            'I1': 0.013, 'I2': 0.013, 'I3': 0.013,
            'b1': 0.1, 'b2': 0.1, 'b3': 0.1,
            'g': 9.81
        }
    
    def get_matrices(self):
        # Simplified 3-link model linearized at horizontal
        m1, m2, m3 = self.params['m1'], self.params['m2'], self.params['m3']
        l1, l2, l3 = self.params['l1'], self.params['l2'], self.params['l3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        g = self.params['g']
        
        # Approximate inertia matrix
        M = jnp.diag(jnp.array([m1*l1**2 + m2*(l1**2 + l2**2) + m3*(l1**2 + l2**2 + l3**2),
                     m2*l2**2 + m3*(l2**2 + l3**2),
                     m3*l3**2]))
        
        # Damping
        D = jnp.diag(jnp.array([b1, b2, b3]))
        
        # Gravity (linearized)
        K = jnp.diag(jnp.array([g*(m1*l1/2 + m2*l1 + m3*l1),
                     g*(m2*l2/2 + m3*l2),
                     g*m3*l3/2]))
        
        Minv = jnp.linalg.inv(M)
        
        A = jnp.zeros((6, 6))
        A = A.at[0:3, 3:6].set(jnp.eye(3))
        A = A.at[3:6, 0:3].set(-Minv @ K)
        A = A.at[3:6, 3:6].set(-Minv @ D)
        
        B = jnp.zeros((6, 3))
        B = B.at[3:6, :].set(Minv)
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))
        R = jnp.diag(jnp.array([1.0, 1.0, 1.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # 3-link arm: each joint ±60° with realistic velocities
        return np.array([
            np.random.uniform(-1.0, 1.0),   # theta1 (rad) ≈ ±60°
            np.random.uniform(-1.0, 1.0),   # theta2 (rad) ≈ ±60°
            np.random.uniform(-1.0, 1.0),   # theta3 (rad) ≈ ±60°
            np.random.uniform(-1.5, 1.5),   # theta1_dot (rad/s)
            np.random.uniform(-1.5, 1.5),   # theta2_dot (rad/s)
            np.random.uniform(-1.5, 1.5)    # theta3_dot (rad/s)
        ])


class UnicycleRobot(LTISystem):
    """
    Unicycle mobile robot (linearized around forward motion)
    State: [x, y, theta, v, omega]
    Input: [acceleration, angular_acceleration]
    """
    
    def get_default_params(self):
        return {
            'v0': 1.0,   # nominal forward velocity (m/s)
            'b_v': 0.1,  # linear damping
            'b_w': 0.1   # angular damping
        }
    
    def get_matrices(self):
        v0 = self.params['v0']
        b_v = self.params['b_v']
        b_w = self.params['b_w']
        
        # Linearized around straight-line motion (theta=0, v=v0, omega=0)
        A = jnp.array([
            [0, 0, 0, 1, 0],
            [0, 0, v0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, -b_v, 0],
            [0, 0, 0, 0, -b_w]
        ])
        B = jnp.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([10.0, 10.0, 5.0, 1.0, 1.0]))
        R = jnp.diag(jnp.array([1.0, 1.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # Unicycle robot: workspace ±3m, heading ±60°, realistic speeds
        return np.array([
            np.random.uniform(-3.0, 3.0),   # x (m)
            np.random.uniform(-3.0, 3.0),   # y (m)
            np.random.uniform(-1.0, 1.0),   # theta (rad) ≈ ±60°
            np.random.uniform(-2.0, 2.0),   # v (m/s) - forward velocity
            np.random.uniform(-1.0, 1.0)    # omega (rad/s) - angular velocity
        ])


class DifferentialDriveRobot(LTISystem):
    """
    Differential drive robot (linearized)
    State: [x, y, theta, v_left, v_right]
    Input: [force_left, force_right]
    """
    
    def get_default_params(self):
        return {
            'm': 5.0,    # robot mass (kg) - reduced
            'I': 0.5,    # moment of inertia (kg·m^2) - reduced
            'r': 0.05,   # wheel radius (m)
            'L': 0.25,   # wheelbase (m) - reduced
            'b': 1.0     # damping - increased
        }
    
    def get_matrices(self):
        m, I, r, L, b = (self.params['m'], self.params['I'], self.params['r'],
                         self.params['L'], self.params['b'])
        
        # Linearized around straight motion with better coupling
        A = jnp.array([
            [0, 0, 0, 0.5*r, 0.5*r],
            [0, 0, 1.0, 0, 0],  # Added coupling from theta to y
            [0, 0, 0, r/L, -r/L],
            [0, 0, 0, -b/m, 0],
            [0, 0, 0, 0, -b/m]
        ])
        B = jnp.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1/m, 0],
            [0, 1/m]
        ])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([20.0, 20.0, 10.0, 1.0, 1.0]))  # Increased position penalties
        R = jnp.diag(jnp.array([10.0, 10.0]))  # Increased R
        return Q, R
    
    def sample_initial_condition(self):
        # Differential drive: workspace ±3m, heading ±60°, wheel speeds realistic
        return np.array([
            np.random.uniform(-3.0, 3.0),   # x (m)
            np.random.uniform(-3.0, 3.0),   # y (m)
            np.random.uniform(-1.0, 1.0),   # theta (rad) ≈ ±60°
            np.random.uniform(-1.0, 1.0),   # v_left (m/s)
            np.random.uniform(-1.0, 1.0)    # v_right (m/s)
        ])


class SCARARobot(LTISystem):
    """
    SCARA robot (Selective Compliance Assembly Robot Arm) - 4 DOF
    State: [theta1, theta2, z, psi, theta1_dot, theta2_dot, z_dot, psi_dot]
    Input: [torque1, torque2, force_z, torque_psi]
    """
    
    def get_default_params(self):
        return {
            'm1': 2.0, 'm2': 1.5, 'm3': 1.0,  # link masses
            'l1': 0.3, 'l2': 0.25,  # link lengths
            'I1': 0.04, 'I2': 0.03, 'I3': 0.01,  # inertias
            'b1': 0.5, 'b2': 0.5, 'b3': 0.3, 'b4': 0.2,  # damping
            'g': 9.81
        }
    
    def get_matrices(self):
        # Linearized SCARA dynamics around horizontal configuration
        m1, m2, m3 = self.params['m1'], self.params['m2'], self.params['m3']
        l1, l2 = self.params['l1'], self.params['l2']
        b1, b2, b3, b4 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4']
        g = self.params['g']
        
        # Simplified inertia matrix
        M1 = m1*l1**2 + m2*(l1**2 + l2**2)
        M2 = m2*l2**2
        
        A = jnp.zeros((8, 8))
        A = A.at[0:4, 4:8].set(jnp.eye(4))
        A = A.at[4:8, 4:8].set(-jnp.diag(jnp.array([b1/M1, b2/M2, b3/m3, b4/self.params['I3']])))
        
        B = jnp.zeros((8, 4))
        B = B.at[4, 0].set(1/M1)
        B = B.at[5, 1].set(1/M2)
        B = B.at[6, 2].set(1/m3)
        B = B.at[7, 3].set(1/self.params['I3'])
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([20.0, 20.0, 30.0, 15.0, 2.0, 2.0, 3.0, 1.5]))
        R = jnp.diag(jnp.array([10.0, 10.0, 10.0, 10.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # SCARA robot: 4-DOF industrial robot with realistic workspace
        return np.array([
            np.random.uniform(-1.0, 1.0),   # theta1 (rad) ≈ ±60°
            np.random.uniform(-1.0, 1.0),   # theta2 (rad) ≈ ±60°
            np.random.uniform(-0.2, 0.2),   # z (m) - vertical position
            np.random.uniform(-1.0, 1.0),   # psi (rad) - tool orientation ≈ ±60°
            np.random.uniform(-1.5, 1.5),   # theta1_dot (rad/s)
            np.random.uniform(-1.5, 1.5),   # theta2_dot (rad/s)
            np.random.uniform(-0.5, 0.5),   # z_dot (m/s)
            np.random.uniform(-1.0, 1.0)    # psi_dot (rad/s)
        ])


class SegwayRobot(LTISystem):
    """
    Segway-like wheeled inverted pendulum
    State: [x, theta, x_dot, theta_dot]
    Input: [torque]
    """
    
    def get_default_params(self):
        return {
            'M': 50.0,   # body mass (kg)
            'm': 5.0,    # wheel mass (kg)
            'L': 0.8,    # body length to COM (m)
            'r': 0.15,   # wheel radius (m)
            'Ib': 10.0,  # body inertia
            'Iw': 0.1,   # wheel inertia
            'b': 0.5,    # friction
            'g': 9.81
        }
    
    def get_matrices(self):
        M, m, L, r, Ib, Iw, b, g = (
            self.params['M'], self.params['m'], self.params['L'], self.params['r'],
            self.params['Ib'], self.params['Iw'], self.params['b'], self.params['g']
        )
        
        # Total moment of inertia
        I_total = Ib + M*L**2
        
        # Linearized around upright
        A = jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -M*g*L/(M+m), -b/(M+m), 0],
            [0, (M+m)*g*L/(I_total*(M+m)), b*L/(I_total*(M+m)), 0]
        ])
        B = jnp.array([[0], [0], [1/(r*(M+m))], [-L/(r*I_total*(M+m))]])
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([10.0, 100.0, 10.0, 10.0]))  # High penalty on tilt
        R = jnp.array([[10.0]])  # Very high R to reduce huge control signals (was 50.0)
        return Q, R
    
    def sample_initial_condition(self):
        # Segway: balancing robot, position ±2m, small tilt ±20°, realistic velocities
        return np.array([
            np.random.uniform(-2.0, 2.0),   # x (m) - position
            np.random.uniform(-0.35, 0.35), # theta (rad) ≈ ±20° - must stay small for stability
            np.random.uniform(-1.5, 1.5),   # x_dot (m/s)
            np.random.uniform(-1.0, 1.0)    # theta_dot (rad/s)
        ])


class OmnidirectionalRobot(LTISystem):
    """
    Omnidirectional robot with 3 wheels (holonomic)
    State: [x, y, theta, v_x, v_y, omega]
    Input: [f_x, f_y, torque]
    """
    
    def get_default_params(self):
        return {
            'm': 8.0,    # mass (kg)
            'I': 1.0,    # moment of inertia
            'b_v': 1.0,  # linear damping
            'b_w': 0.5   # angular damping
        }
    
    def get_matrices(self):
        m, I, b_v, b_w = (self.params['m'], self.params['I'], 
                          self.params['b_v'], self.params['b_w'])
        
        A = jnp.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, -b_v/m, 0, 0],
            [0, 0, 0, 0, -b_v/m, 0],
            [0, 0, 0, 0, 0, -b_w/I]
        ])
        B = jnp.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1/m, 0, 0],
            [0, 1/m, 0],
            [0, 0, 1/I]
        ])
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([20.0, 20.0, 10.0, 2.0, 2.0, 1.0]))
        R = jnp.diag(jnp.array([10.0, 10.0, 10.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # Omnidirectional robot: holonomic, workspace ±3m, heading ±60°
        return np.array([
            np.random.uniform(-3.0, 3.0),   # x (m)
            np.random.uniform(-3.0, 3.0),   # y (m)
            np.random.uniform(-1.0, 1.0),   # theta (rad) ≈ ±60°
            np.random.uniform(-2.0, 2.0),   # v_x (m/s)
            np.random.uniform(-2.0, 2.0),   # v_y (m/s)
            np.random.uniform(-1.0, 1.0)    # omega (rad/s)
        ])


class PlanarQuadruped(LTISystem):
    """
    Simplified planar quadruped (reduced complexity, body + legs as single system)
    State: [x, z, theta, x_dot, z_dot, theta_dot]
    Input: [horizontal_force, vertical_force]
    """
    
    def get_default_params(self):
        return {
            'm': 12.0,   # total mass
            'I': 1.5,    # body inertia
            'b_x': 3.0,  # horizontal damping
            'b_z': 3.0,  # vertical damping
            'b_theta': 1.0,  # rotational damping
            'g': 9.81
        }
    
    def get_matrices(self):
        m, I, b_x, b_z, b_theta, g = (
            self.params['m'], self.params['I'], self.params['b_x'],
            self.params['b_z'], self.params['b_theta'], self.params['g']
        )
        
        # Simplified model - body with damping
        A = jnp.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, -b_x/m, 0, 0],
            [0, 0, 0, 0, -b_z/m, 0],
            [0, 0, 0, 0, 0, -b_theta/I]
        ])
        
        B = jnp.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1/m, 0],
            [0, 1/m],
            [0, 0]
        ])
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([20.0, 30.0, 40.0, 2.0, 3.0, 4.0]))
        R = jnp.diag(jnp.array([1.0, 1.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # Planar quadruped (simplified): position ±2m, attitude ±30°
        return np.array([
            np.random.uniform(-2.0, 2.0),   # x (m)
            np.random.uniform(-0.5, 0.5),   # z (m) - vertical position
            np.random.uniform(-0.5, 0.5),   # theta (rad) - body pitch ≈ ±30°
            np.random.uniform(-1.5, 1.5),   # x_dot (m/s)
            np.random.uniform(-1.0, 1.0),   # z_dot (m/s)
            np.random.uniform(-1.0, 1.0)    # theta_dot (rad/s)
        ])


class CableDrivenRobot(LTISystem):
    """
    Cable-driven parallel robot (2D, 3 cables)
    State: [x, y, x_dot, y_dot]
    Input: [tension_deviation1, tension_deviation2, tension_deviation3]
    """
    
    def get_default_params(self):
        return {
            'm': 2.0,    # end-effector mass
            'b': 0.5,    # damping
            'g': 9.81
        }
    
    def get_matrices(self):
        m, b, g = self.params['m'], self.params['b'], self.params['g']
        
        # Linearized around center position with equal tensions
        # Cable geometry (simplified)
        A = jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, -b/m, 0],
            [0, 0, 0, -b/m]
        ])
        
        # Force transformation matrix (simplified)
        B = jnp.array([
            [0, 0, 0],
            [0, 0, 0],
            [0.5/m, -0.25/m, -0.25/m],
            [0, 0.433/m, -0.433/m]
        ])
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([50.0, 50.0, 5.0, 5.0]))
        R = jnp.diag(jnp.array([10.0, 10.0, 10.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # Cable-driven robot: workspace ±1m in 2D plane
        return np.array([
            np.random.uniform(-1.0, 1.0),   # x (m)
            np.random.uniform(-1.0, 1.0),   # y (m)
            np.random.uniform(-1.0, 1.0),   # x_dot (m/s)
            np.random.uniform(-1.0, 1.0)    # y_dot (m/s)
        ])


class FlexibleJointRobot(LTISystem):
    """
    Single-link robot with flexible joint
    State: [theta_link, theta_motor, theta_link_dot, theta_motor_dot]
    Input: [motor_torque]
    """
    
    def get_default_params(self):
        return {
            'Jl': 0.5,   # link inertia
            'Jm': 0.05,  # motor inertia
            'k': 100.0,  # joint stiffness
            'bl': 0.5,   # link damping
            'bm': 0.1,   # motor damping
            'mgl': 5.0   # gravity torque coefficient
        }
    
    def get_matrices(self):
        Jl, Jm, k, bl, bm, mgl = (
            self.params['Jl'], self.params['Jm'], self.params['k'],
            self.params['bl'], self.params['bm'], self.params['mgl']
        )
        
        A = jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-k/Jl - mgl/Jl, k/Jl, -bl/Jl, 0],
            [k/Jm, -k/Jm, 0, -bm/Jm]
        ])
        B = jnp.array([[0], [0], [0], [1/Jm]])
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([100.0, 50.0, 100.0, 50.0]))
        R = jnp.array([[100.0]])  # Extremely high R to prevent huge control (was 100)
        return Q, R
    
    def sample_initial_condition(self):
        # Flexible joint robot: link and motor angles can differ, ±60°
        return np.array([
            np.random.uniform(-1.0, 1.0),   # theta_link (rad) ≈ ±60°
            np.random.uniform(-1.0, 1.0),   # theta_motor (rad) ≈ ±60°
            np.random.uniform(-2.0, 2.0),   # theta_link_dot (rad/s)
            np.random.uniform(-2.0, 2.0)    # theta_motor_dot (rad/s)
        ])


class PlanarBiped(LTISystem):
    """
    Simplified planar biped (inverted pendulum on cart with leg dynamics)
    State: [x, theta_torso, x_dot, theta_torso_dot]
    Input: [ankle_torque]
    """
    
    def get_default_params(self):
        return {
            'm': 20.0,   # total mass
            'L': 0.9,    # height to COM
            'I': 2.0,    # torso inertia
            'b_x': 1.0,  # horizontal damping
            'b_theta': 0.5,  # angular damping
            'g': 9.81
        }
    
    def get_matrices(self):
        m, L, I, b_x, b_theta, g = (
            self.params['m'], self.params['L'], self.params['I'],
            self.params['b_x'], self.params['b_theta'], self.params['g']
        )
        
        # Similar to inverted pendulum on cart
        I_total = I + m*L**2
        
        A = jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -m*g*L/m, -b_x/m, b_theta/(m*L)],
            [0, m*g*L/I_total, b_x*L/I_total, -b_theta/I_total]
        ])
        B = jnp.array([[0], [0], [1/m], [-L/I_total]])
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([10.0, 200.0, 10.0, 20.0]))  # Very high penalty on tilt
        R = jnp.array([[50.0]])  # Higher R to prevent large control (was 10)
        return Q, R
    
    def sample_initial_condition(self):
        # Planar biped: balancing, position ±1m, small tilt ±20°
        return np.array([
            np.random.uniform(-1.0, 1.0),   # x (m)
            np.random.uniform(-0.35, 0.35), # theta_torso (rad) ≈ ±20° - balancing
            np.random.uniform(-1.5, 1.5),   # x_dot (m/s)
            np.random.uniform(-1.0, 1.0)    # theta_torso_dot (rad/s)
        ])


class SixDOFManipulator(LTISystem):
    """
    6-DOF robotic manipulator (simplified, linearized)
    State: [q1, q2, q3, q4, q5, q6, q1_dot, q2_dot, q3_dot, q4_dot, q5_dot, q6_dot]
    Input: [tau1, tau2, tau3, tau4, tau5, tau6]
    """
    
    def get_default_params(self):
        return {
            'm': [2.0, 1.8, 1.5, 1.0, 0.8, 0.5],  # link masses
            'l': [0.4, 0.35, 0.3, 0.25, 0.2, 0.15],  # link lengths
            'I': [0.05, 0.04, 0.03, 0.02, 0.015, 0.01],  # inertias
            'b': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],  # damping
            'g': 9.81
        }
    
    def get_matrices(self):
        m = self.params['m']
        l = self.params['l']
        I = self.params['I']
        b = self.params['b']
        g = self.params['g']
        
        # Approximate inertia matrix (diagonal approximation)
        M_diag = [I[i] + sum([m[j]*l[i]**2 for j in range(i+1, 6)]) for i in range(6)]
        
        # Linearized around horizontal configuration
        A = jnp.zeros((12, 12))
        A = A.at[0:6, 6:12].set(jnp.eye(6))
        
        # Damping and gravity terms
        for i in range(6):
            A = A.at[6+i, i].set(-g * sum([m[j]*l[i]/2 for j in range(i, 6)]) / M_diag[i])
            A = A.at[6+i, 6+i].set(-b[i] / M_diag[i])
        
        B = jnp.zeros((12, 6))
        for i in range(6):
            B = B.at[6+i, i].set(1/M_diag[i])
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([15.0, 15.0, 15.0, 10.0, 10.0, 10.0, 
                     2.0, 2.0, 2.0, 1.0, 1.0, 1.0]))
        R = jnp.diag(jnp.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # 6-DOF manipulator: full workspace, each joint ±60°
        return np.array([
            np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5),
            np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5),
            np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)
        ])


class DualArmRobot(LTISystem):
    """
    Dual-arm robot (simplified, two 2-link arms)
    State: [q1L, q2L, q1R, q2R, q1L_dot, q2L_dot, q1R_dot, q2R_dot]
    Input: [tau1L, tau2L, tau1R, tau2R]
    """
    
    def get_default_params(self):
        return {
            'm1': 1.5, 'm2': 1.0,  # link masses (per arm)
            'l1': 0.4, 'l2': 0.35,  # link lengths
            'I1': 0.03, 'I2': 0.02,  # inertias
            'b': 0.6,  # damping
            'g': 9.81
        }
    
    def get_matrices(self):
        m1, m2, l1, l2, I1, I2, b, g = (
            self.params['m1'], self.params['m2'], self.params['l1'], self.params['l2'],
            self.params['I1'], self.params['I2'], self.params['b'], self.params['g']
        )
        
        # Each arm has similar dynamics (decoupled approximation)
        M11 = I1 + I2 + m2*l1**2
        M22 = I2
        
        # Build matrices for dual arm
        A = jnp.zeros((8, 8))
        A = A.at[0:4, 4:8].set(jnp.eye(4))
        
        # Left arm dynamics
        A = A.at[4, 0].set(-g*(m1*l1/2 + m2*l1)/M11)
        A = A.at[4, 4].set(-b/M11)
        A = A.at[5, 1].set(-g*m2*l2/(2*M22))
        A = A.at[5, 5].set(-b/M22)
        
        # Right arm dynamics (symmetric)
        A = A.at[6, 2].set(-g*(m1*l1/2 + m2*l1)/M11)
        A = A.at[6, 6].set(-b/M11)
        A = A.at[7, 3].set(-g*m2*l2/(2*M22))
        A = A.at[7, 7].set(-b/M22)
        
        B = jnp.zeros((8, 4))
        B = B.at[4, 0].set(1/M11)
        B = B.at[5, 1].set(1/M22)
        B = B.at[6, 2].set(1/M11)
        B = B.at[7, 3].set(1/M22)
        
        return A, B
    
    def get_default_lqr_weights(self):
        Q = jnp.diag(jnp.array([15.0, 15.0, 15.0, 15.0, 2.0, 2.0, 2.0, 2.0]))
        R = jnp.diag(jnp.array([10.0, 10.0, 10.0, 10.0]))
        return Q, R
    
    def sample_initial_condition(self):
        # Dual-arm robot: two 2-link arms, each joint ±60°
        return np.array([
            np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0),  # Left arm angles
            np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0),  # Right arm angles
            np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5),  # Left arm velocities
            np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)   # Right arm velocities
        ])

