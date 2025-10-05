"""
Mechanical LTI Systems
"""

import numpy as np
from .base_system import LTISystem


class MassSpringDamper(LTISystem):
    """
    Mass-spring-damper system: m*x'' + c*x' + k*x = F
    State: [position, velocity]
    Input: [force]
    """
    
    def get_default_params(self):
        return {
            'm': 1.0,   # mass (kg)
            'c': 0.5,   # damping coefficient (N·s/m)
            'k': 2.0    # spring constant (N/m)
        }
    
    def get_matrices(self):
        m, c, k = self.params['m'], self.params['c'], self.params['k']
        A = np.array([
            [0, 1],
            [-k/m, -c/m]
        ])
        B = np.array([[0], [1/m]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 1.0])  # Penalize position more
        R = np.array([[0.1]])
        return Q, R
    
    def sample_initial_condition(self):
        # Realistic for a mechanical system with ~1m natural length
        # Position: ±2m displacement, Velocity: ±2 m/s
        return np.array([
            np.random.uniform(-2.0, 2.0),  # position (m)
            np.random.uniform(-2.0, 2.0)   # velocity (m/s)
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([1.0, 0.5])


class SimplePendulum(LTISystem):
    """
    Simple pendulum (linearized around downward equilibrium)
    State: [angle, angular_velocity]
    Input: [torque]
    """
    
    def get_default_params(self):
        return {
            'm': 1.0,   # mass (kg)
            'l': 1.0,   # length (m)
            'b': 0.1,   # damping (N·m·s)
            'g': 9.81   # gravity (m/s^2)
        }
    
    def get_matrices(self):
        m, l, b, g = self.params['m'], self.params['l'], self.params['b'], self.params['g']
        I = m * l**2  # moment of inertia
        A = np.array([
            [0, 1],
            [g/l, -b/I]
        ])
        B = np.array([[0], [1/I]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([20.0, 1.0])
        R = np.array([[0.1]])
        return Q, R
    
    def sample_initial_condition(self):
        # Pendulum can start from ±60° (1 rad) with realistic angular velocity
        return np.array([
            np.random.uniform(-1.0, 1.0),   # angle (rad) ≈ ±60°
            np.random.uniform(-2.0, 2.0)    # angular velocity (rad/s)
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([0.3, 0.2])


class InvertedPendulum(LTISystem):
    """
    Inverted pendulum (linearized around upward equilibrium)
    State: [angle, angular_velocity]
    Input: [torque]
    """
    
    def get_default_params(self):
        return {
            'm': 0.5,   # mass (kg)
            'l': 0.5,   # length (m)
            'b': 0.05,  # damping (N·m·s)
            'g': 9.81   # gravity (m/s^2)
        }
    
    def get_matrices(self):
        m, l, b, g = self.params['m'], self.params['l'], self.params['b'], self.params['g']
        I = m * l**2
        A = np.array([
            [0, 1],
            [g/l, -b/I]  # Positive g/l makes it unstable
        ])
        B = np.array([[0], [1/I]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([50.0, 5.0])  # High penalty on angle
        R = np.array([[0.01]])
        return Q, R
    
    def sample_initial_condition(self):
        # Inverted pendulum: smaller angles to stay in linear region
        # But still realistic perturbations
        return np.array([
            np.random.uniform(-0.4, 0.4),   # angle (rad) ≈ ±23°
            np.random.uniform(-1.0, 1.0)    # angular velocity (rad/s)
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([0.2, 0.1])


class DoublePendulum(LTISystem):
    """
    Double pendulum (linearized, simplified 4-state model)
    State: [theta1, theta2, theta1_dot, theta2_dot]
    Input: [torque on joint 1]
    """
    
    def get_default_params(self):
        return {
            'm1': 1.0,  # mass 1 (kg)
            'm2': 1.0,  # mass 2 (kg)
            'l1': 1.0,  # length 1 (m)
            'l2': 1.0,  # length 2 (m)
            'b1': 0.1,  # damping 1
            'b2': 0.1,  # damping 2
            'g': 9.81
        }
    
    def get_matrices(self):
        # Simplified linearized model
        m1, m2 = self.params['m1'], self.params['m2']
        l1, l2 = self.params['l1'], self.params['l2']
        b1, b2 = self.params['b1'], self.params['b2']
        g = self.params['g']
        
        # Linearized around downward position
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [g*(m1+m2)/(m1*l1), -g*m2/(m1*l1), -b1/(m1*l1**2), 0],
            [-g*(m1+m2)/(m2*l2), g*(m1+m2)/(m2*l2), 0, -b2/(m2*l2**2)]
        ])
        B = np.array([[0], [0], [1/(m1*l1**2)], [0]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 10.0, 1.0, 1.0])
        R = np.array([[0.1]])
        return Q, R
    
    def sample_initial_condition(self):
        # Double pendulum: each link can start at ±45° with realistic velocities
        return np.array([
            np.random.uniform(-0.8, 0.8),   # theta1 (rad) ≈ ±45°
            np.random.uniform(-0.8, 0.8),   # theta2 (rad) ≈ ±45°
            np.random.uniform(-1.5, 1.5),   # theta1_dot (rad/s)
            np.random.uniform(-1.5, 1.5)    # theta2_dot (rad/s)
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([0.3, 0.3, 0.2, 0.2])


class CartPole(LTISystem):
    """
    Cart-pole system (linearized around upward pendulum)
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Input: [force on cart]
    """
    
    def get_default_params(self):
        return {
            'M': 1.0,   # cart mass (kg)
            'm': 0.3,   # pendulum mass (kg)
            'l': 0.5,   # pendulum length (m)
            'b': 0.1,   # cart friction
            'g': 9.81
        }
    
    def get_matrices(self):
        M, m, l, b, g = (self.params['M'], self.params['m'], 
                         self.params['l'], self.params['b'], self.params['g'])
        
        I = m * l**2
        total_mass = M + m
        
        A = np.array([
            [0, 1, 0, 0],
            [0, -b/M, -m*g/M, 0],
            [0, 0, 0, 1],
            [0, b/(M*l), (M+m)*g/(M*l), 0]
        ])
        B = np.array([[0], [1/M], [0], [-1/(M*l)]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 1.0, 100.0, 10.0])  # High penalty on pole angle
        R = np.array([[0.01]])
        return Q, R
    
    def sample_initial_condition(self):
        # Cart-pole: track ±2m, pendulum ±30°, realistic velocities
        return np.array([
            np.random.uniform(-2.0, 2.0),   # x (m) - cart position
            np.random.uniform(-1.0, 1.0),   # x_dot (m/s) - cart velocity
            np.random.uniform(-0.5, 0.5),   # theta (rad) - pole angle ≈ ±30°
            np.random.uniform(-1.0, 1.0)    # theta_dot (rad/s) - angular velocity
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([0.5, 0.2, 0.15, 0.1])


class Acrobot(LTISystem):
    """
    Acrobot (two-link underactuated robot, linearized)
    State: [theta1, theta2, theta1_dot, theta2_dot]
    Input: [torque on joint 2]
    """
    
    def get_default_params(self):
        return {
            'm1': 1.0,
            'm2': 1.0,
            'l1': 1.0,
            'l2': 1.0,
            'lc1': 0.5,  # center of mass link 1
            'lc2': 0.5,  # center of mass link 2
            'I1': 0.083,  # inertia link 1
            'I2': 0.083,  # inertia link 2
            'g': 9.81
        }
    
    def get_matrices(self):
        # Simplified linearized model around downward position
        m1, m2 = self.params['m1'], self.params['m2']
        l1, l2 = self.params['l1'], self.params['l2']
        g = self.params['g']
        
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [g*(m1+m2)/l1, -g*m2/l1, 0, 0],
            [-g/l2, g/l2, 0, 0]
        ])
        B = np.array([[0], [0], [0], [1/(m2*l2**2)]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([15.0, 15.0, 2.0, 2.0])
        R = np.array([[0.1]])
        return Q, R
    
    def sample_initial_condition(self):
        # Acrobot: both links can swing ±60°
        return np.array([
            np.random.uniform(-1.0, 1.0),   # theta1 (rad) ≈ ±60°
            np.random.uniform(-1.0, 1.0),   # theta2 (rad) ≈ ±60°
            np.random.uniform(-2.0, 2.0),   # theta1_dot (rad/s)
            np.random.uniform(-2.0, 2.0)    # theta2_dot (rad/s)
        ])


class FurutaPendulum(LTISystem):
    """
    Furuta pendulum (rotary inverted pendulum)
    State: [arm_angle, pendulum_angle, arm_velocity, pendulum_velocity]
    Input: [torque on arm]
    """
    
    def get_default_params(self):
        return {
            'Jr': 0.05,   # arm inertia (reduced)
            'Jp': 0.01,   # pendulum inertia
            'mp': 0.15,   # pendulum mass (reduced)
            'lp': 0.2,    # pendulum length (reduced)
            'lr': 0.15,   # arm length (reduced)
            'br': 0.02,   # arm damping (increased)
            'bp': 0.005,  # pendulum damping (increased)
            'g': 9.81
        }
    
    def get_matrices(self):
        Jr, Jp, mp, lp, lr, br, bp, g = (
            self.params['Jr'], self.params['Jp'], self.params['mp'],
            self.params['lp'], self.params['lr'], self.params['br'],
            self.params['bp'], self.params['g']
        )
        
        # Linearized around upward pendulum (improved model)
        # Total moment of inertia
        J_total = Jr + mp * lr**2
        
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, -br/J_total, 0],
            [0, mp*g*lp/Jp, 0, -bp/Jp]
        ])
        B = np.array([[0], [0], [1/J_total], [lr/(Jp*lp)]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([1.0, 100.0, 1.0, 10.0])  # Very high penalty on pendulum angle
        R = np.array([[1.0]])  # Higher R to prevent excessive control
        return Q, R
    
    def sample_initial_condition(self):
        # Furuta pendulum: arm ±60°, pendulum ±30° (staying near upright)
        return np.array([
            np.random.uniform(-1.0, 1.0),   # theta_arm (rad) ≈ ±60°
            np.random.uniform(-0.5, 0.5),   # theta_pendulum (rad) ≈ ±30°
            np.random.uniform(-1.5, 1.5),   # theta_arm_dot (rad/s)
            np.random.uniform(-1.0, 1.0)    # theta_pendulum_dot (rad/s)
        ])


class BallAndBeam(LTISystem):
    """
    Ball and beam system
    State: [ball_position, ball_velocity, beam_angle, beam_angular_velocity]
    Input: [beam torque]
    """
    
    def get_default_params(self):
        return {
            'm': 0.05,   # ball mass
            'R': 0.02,   # ball radius
            'Jb': 2e-6,  # ball inertia
            'L': 0.5,    # beam length
            'g': 9.81
        }
    
    def get_matrices(self):
        m, R, Jb, L, g = (self.params['m'], self.params['R'], 
                          self.params['Jb'], self.params['L'], self.params['g'])
        
        J_total = Jb + m*R**2
        
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -m*g/J_total*R**2, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        B = np.array([[0], [0], [0], [1]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([20.0, 2.0, 10.0, 1.0])
        R = np.array([[0.1]])
        return Q, R
    
    def sample_initial_condition(self):
        # Ball-and-beam: beam ±30°, ball position ±0.5m on beam
        return np.array([
            np.random.uniform(-0.5, 0.5),   # ball position (m) along beam
            np.random.uniform(-0.5, 0.5),   # beam angle (rad) ≈ ±30°
            np.random.uniform(-0.5, 0.5),   # ball velocity (m/s)
            np.random.uniform(-0.5, 0.5)    # beam angular velocity (rad/s)
        ])


class BallAndPlate(LTISystem):
    """
    Ball and plate system (2D)
    State: [x, y, x_dot, y_dot, theta_x, theta_y]
    Input: [torque_x, torque_y]
    """
    
    def get_default_params(self):
        return {
            'm': 0.05,
            'R': 0.02,
            'g': 9.81,
            'Jb': 2e-6
        }
    
    def get_matrices(self):
        m, R, g, Jb = self.params['m'], self.params['R'], self.params['g'], self.params['Jb']
        J_total = Jb + m*R**2
        
        A = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, -m*g/J_total*R**2, 0],
            [0, 0, 0, 0, 0, -m*g/J_total*R**2],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([20.0, 20.0, 2.0, 2.0, 10.0, 10.0])
        R = np.diag([0.1, 0.1])
        return Q, R
    
    def sample_initial_condition(self):
        # Ball-and-plate: ball position ±0.3m on plate, angles ±30°
        return np.array([
            np.random.uniform(-0.3, 0.3),   # ball x position (m)
            np.random.uniform(-0.3, 0.3),   # ball y position (m)
            np.random.uniform(-0.5, 0.5),   # plate angle x (rad) ≈ ±30°
            np.random.uniform(-0.5, 0.5),   # plate angle y (rad) ≈ ±30°
            np.random.uniform(-0.5, 0.5),   # ball x velocity (m/s)
            np.random.uniform(-0.5, 0.5)    # ball y velocity (m/s)
        ])


class ReactionWheelPendulum(LTISystem):
    """
    Reaction wheel pendulum (inertia wheel pendulum)
    State: [pendulum_angle, wheel_angle, pendulum_velocity, wheel_velocity]
    Input: [wheel_torque]
    """
    
    def get_default_params(self):
        return {
            'mp': 0.3,   # pendulum mass (reduced)
            'mw': 0.1,   # wheel mass (reduced)
            'lp': 0.25,  # pendulum length (reduced)
            'Jp': 0.008, # pendulum inertia (reduced)
            'Jw': 0.001, # wheel inertia
            'bp': 0.02,  # pendulum damping (increased)
            'bw': 0.005, # wheel damping (increased)
            'g': 9.81
        }
    
    def get_matrices(self):
        mp, lp, Jp, Jw, bp, bw, g = (
            self.params['mp'], self.params['lp'], self.params['Jp'],
            self.params['Jw'], self.params['bp'], self.params['bw'], self.params['g']
        )
        
        # Improved coupling model
        J_total = Jp + Jw
        
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [mp*g*lp/Jp, 0, -bp/Jp, bw/Jp],
            [-mp*g*lp/Jw, 0, bp/Jw, -bw/Jw]
        ])
        B = np.array([[0], [0], [-1/Jp], [(Jp+Jw)/(Jp*Jw)]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([100.0, 0.1, 10.0, 0.1])  # High penalty on pendulum angle
        R = np.array([[1.0]])  # Higher R
        return Q, R
    
    def sample_initial_condition(self):
        # Reaction wheel pendulum: pendulum ±45°, wheel ±120° (2 rad)
        return np.array([
            np.random.uniform(-0.8, 0.8),   # pendulum angle (rad) ≈ ±45°
            np.random.uniform(-2.0, 2.0),   # wheel angle (rad) ≈ ±115°
            np.random.uniform(-1.0, 1.0),   # pendulum angular velocity (rad/s)
            np.random.uniform(-2.0, 2.0)    # wheel angular velocity (rad/s)
        ])


class FlexibleBeam(LTISystem):
    """
    Flexible beam (Euler-Bernoulli, first mode approximation)
    State: [tip_deflection, tip_velocity]
    Input: [force at tip]
    """
    
    def get_default_params(self):
        return {
            'E': 5e8,     # Young's modulus (Pa) - much softer beam
            'I': 1e-8,    # second moment of area (m^4)
            'L': 0.4,     # length (m) - longer for lower frequency
            'm': 0.1,     # mass (kg) - heavier for lower frequency
            'c': 10.0     # damping - very high
        }
    
    def get_matrices(self):
        E, I, L, m, c = (self.params['E'], self.params['I'], 
                         self.params['L'], self.params['m'], self.params['c'])
        
        # Natural frequency of first mode
        omega_n = (1.875**2) * np.sqrt(E*I / (m*L**4))
        
        # Heavily damped system to prevent stiffness
        zeta = 2.0  # Overdamped (zeta > 1)
        A = np.array([
            [0, 1],
            [-omega_n**2, -2*zeta*omega_n]  # Overdamped system
        ])
        B = np.array([[0], [1/m]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([1000.0, 10.0])  # Higher position penalty
        R = np.array([[100.0]])  # Very high R to prevent aggressive control
        return Q, R
    
    def sample_initial_condition(self):
        # Flexible beam: tip deflection ±0.1m, velocity ±0.5 m/s
        return np.array([
            np.random.uniform(-0.1, 0.1),   # tip deflection (m)
            np.random.uniform(-0.5, 0.5)    # tip velocity (m/s)
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([0.01, 0.05])


class MagneticLevitation(LTISystem):
    """
    Magnetic levitation system (linearized)
    State: [position, velocity, current]
    Input: [voltage]
    """
    
    def get_default_params(self):
        return {
            'm': 0.02,   # ball mass (kg) - lighter
            'k': 0.01,   # magnetic constant - reduced
            'L': 0.1,    # inductance (H) - increased
            'R': 10.0,   # resistance (Ohm) - increased
            'g': 9.81
        }
    
    def get_matrices(self):
        m, k, L, R, g = (self.params['m'], self.params['k'], 
                         self.params['L'], self.params['R'], self.params['g'])
        
        # Linearized around equilibrium
        i_eq = np.sqrt(m * g / k)
        z_eq = 0.02  # equilibrium position - higher
        
        # More conservative linearization
        A = np.array([
            [0, 1, 0],
            [g/z_eq, -0.1, -k*i_eq/(m*z_eq)],  # Simplified and stabilized
            [0, 0, -R/L]
        ])
        B = np.array([[0], [0], [1/L]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([100.0, 10.0, 0.1])  # Higher position penalty, lower current
        R = np.array([[10.0]])  # Much higher R
        return Q, R
    
    def sample_initial_condition(self):
        # Magnetic levitation: position ±0.02m from equilibrium, velocity ±0.1 m/s, current ±2A
        return np.array([
            np.random.uniform(-0.02, 0.02),  # position deviation (m) - small for stability
            np.random.uniform(-0.1, 0.1),    # velocity (m/s)
            np.random.uniform(-2.0, 2.0)     # current (A)
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([0.005, 0.01, 0.5])


class SuspensionSystem(LTISystem):
    """
    Vehicle suspension system (quarter-car model)
    State: [sprung_mass_pos, unsprung_mass_pos, sprung_velocity, unsprung_velocity]
    Input: [suspension_force]
    """
    
    def get_default_params(self):
        return {
            'ms': 250.0,  # sprung mass (kg)
            'mu': 40.0,   # unsprung mass (kg) - reduced
            'ks': 10000,  # suspension spring (N/m) - reduced
            'kt': 150000, # tire spring (N/m) - reduced
            'cs': 1500    # suspension damping (N·s/m) - increased
        }
    
    def get_matrices(self):
        ms, mu, ks, kt, cs = (self.params['ms'], self.params['mu'], 
                              self.params['ks'], self.params['kt'], self.params['cs'])
        
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-ks/ms, ks/ms, -cs/ms, cs/ms],
            [ks/mu, -(ks+kt)/mu, cs/mu, -cs/mu]
        ])
        B = np.array([[0], [0], [1/ms], [-1/mu]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([100.0, 10.0, 10.0, 1.0])  # Higher penalties
        R = np.array([[1.0]])  # Much higher R
        return Q, R
    
    def sample_initial_condition(self):
        # Suspension: body displacement ±0.15m, wheel displacement ±0.1m, realistic velocities
        return np.array([
            np.random.uniform(-0.15, 0.15),  # body displacement (m)
            np.random.uniform(-0.1, 0.1),    # wheel displacement (m)
            np.random.uniform(-1.0, 1.0),    # body velocity (m/s)
            np.random.uniform(-2.0, 2.0)     # wheel velocity (m/s)
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([0.05, 0.02, 0.2, 0.5])

