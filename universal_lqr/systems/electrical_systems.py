"""
Electrical LTI Systems
"""

import numpy as np
from .base_system import LTISystem


class DCMotor(LTISystem):
    """
    DC motor (position/speed control)
    State: [angle, angular_velocity, current]
    Input: [voltage]
    """
    
    def get_default_params(self):
        return {
            'J': 0.01,    # rotor inertia (kg·m^2)
            'b': 0.1,     # viscous friction (N·m·s)
            'Kt': 0.01,   # torque constant (N·m/A)
            'Ke': 0.01,   # back-emf constant (V·s/rad)
            'R': 1.0,     # armature resistance (Ohm)
            'L': 0.5      # armature inductance (H)
        }
    
    def get_matrices(self):
        J, b, Kt, Ke, R, L = (self.params['J'], self.params['b'], self.params['Kt'],
                               self.params['Ke'], self.params['R'], self.params['L'])
        
        A = np.array([
            [0, 1, 0],
            [0, -b/J, Kt/J],
            [0, -Ke/L, -R/L]
        ])
        B = np.array([[0], [0], [1/L]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 1.0, 0.1])
        R = np.array([[0.1]])
        return Q, R
    
    def sample_initial_condition(self):
        # DC Motor: multiple rotations, typical speeds, current ±2A
        return np.array([
            np.random.uniform(-2*np.pi, 2*np.pi),  # theta (rad) - multiple turns
            np.random.uniform(-10.0, 10.0),        # omega (rad/s) - realistic motor speed
            np.random.uniform(-2.0, 2.0)           # current (A)
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([1.0, 2.0, 0.5])


class ACMotor(LTISystem):
    """
    AC motor (linearized synchronous machine model in d-q frame)
    State: [id, iq, omega]
    Input: [vd, vq]
    """
    
    def get_default_params(self):
        return {
            'Rs': 2.0,    # stator resistance (Ohm) - increased
            'Ld': 0.05,   # d-axis inductance (H) - increased
            'Lq': 0.05,   # q-axis inductance (H) - increased
            'J': 0.05,    # inertia (kg·m^2) - increased
            'B': 0.5,     # damping (N·m·s) - increased
            'p': 2,       # pole pairs
            'lambda_m': 0.05  # permanent magnet flux linkage - reduced
        }
    
    def get_matrices(self):
        Rs, Ld, Lq, J, B, p, lm = (self.params['Rs'], self.params['Ld'], self.params['Lq'],
                                    self.params['J'], self.params['B'], self.params['p'], 
                                    self.params['lambda_m'])
        
        # Linearized around operating point
        omega_0 = 10  # nominal speed (rad/s) - much reduced
        
        A = np.array([
            [-Rs/Ld, omega_0*Lq/Ld, 0],
            [-omega_0*Ld/Lq, -Rs/Lq, -p*lm/Lq],
            [0, 0.5*p*lm/J, -B/J]  # Reduced coupling
        ])
        B = np.array([
            [1/Ld, 0],
            [0, 1/Lq],
            [0, 0]
        ])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 10.0, 50.0])  # Higher penalties
        R = np.diag([10.0, 10.0])  # Much higher R
        return Q, R
    
    def sample_initial_condition(self):
        return np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-10.0, 10.0)
        ])


class BuckConverter(LTISystem):
    """
    Buck DC-DC converter (averaged model)
    State: [inductor_current, capacitor_voltage]
    Input: [duty_cycle (linearized)]
    """
    
    def get_default_params(self):
        return {
            'L': 50e-3,   # inductance (H) - much larger (50x original)
            'C': 5000e-6, # capacitance (F) - much larger (50x original)
            'R': 10.0,    # load resistance (Ohm)
            'Vin': 12.0   # input voltage (V)
        }
    
    def get_matrices(self):
        L, C, R, Vin = self.params['L'], self.params['C'], self.params['R'], self.params['Vin']
        
        # Linearized around operating point (D=0.5) with ESR
        D = 0.5
        ESR_L = 1.0  # Equivalent series resistance in inductor
        ESR_C = 0.1  # Equivalent series resistance in capacitor
        
        A = np.array([
            [-ESR_L/L, -1/L],  # Added ESR
            [1/C, -(1/(R*C) + ESR_C/C)]  # Added ESR
        ])
        B = np.array([[Vin/L], [0]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([1.0, 1000.0])  # Very high voltage penalty
        R = np.array([[500.0]])  # Much much higher R
        return Q, R
    
    def sample_initial_condition(self):
        return np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-2.0, 2.0)
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([0.5, 2.0])


class BoostConverter(LTISystem):
    """
    Boost DC-DC converter (averaged model)
    State: [inductor_current, capacitor_voltage]
    Input: [duty_cycle (linearized)]
    """
    
    def get_default_params(self):
        return {
            'L': 50e-3,   # inductance (H) - much larger
            'C': 5000e-6, # capacitance (F) - much larger
            'R': 20.0,
            'Vin': 12.0
        }
    
    def get_matrices(self):
        L, C, R, Vin = self.params['L'], self.params['C'], self.params['R'], self.params['Vin']
        
        # Linearized around D=0.3 (more stable) with ESR
        D = 0.3
        ESR_L = 1.0
        ESR_C = 0.1
        
        A = np.array([
            [-ESR_L/L, -(1-D)/L],  # Added ESR
            [(1-D)/C, -(1/(R*C) + ESR_C/C)]  # Added ESR
        ])
        B = np.array([[-Vin/L], [Vin/(R*C)]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([1.0, 1000.0])  # Higher voltage penalty
        R = np.array([[500.0]])  # Much much higher R
        return Q, R
    
    def sample_initial_condition(self):
        return np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-5.0, 5.0)
        ])


class BuckBoostConverter(LTISystem):
    """
    Buck-boost DC-DC converter
    State: [inductor_current, capacitor_voltage]
    Input: [duty_cycle (linearized)]
    """
    
    def get_default_params(self):
        return {
            'L': 50e-3,   # inductance (H) - much larger
            'C': 5000e-6, # capacitance (F) - much larger
            'R': 15.0,
            'Vin': 12.0
        }
    
    def get_matrices(self):
        L, C, R, Vin = self.params['L'], self.params['C'], self.params['R'], self.params['Vin']
        
        # Linearized around D=0.4 (more stable) with ESR
        D = 0.4
        ESR_L = 1.0
        ESR_C = 0.1
        
        A = np.array([
            [-ESR_L/L, -(1-D)/L],  # Added ESR
            [(1-D)/C, -(1/(R*C) + ESR_C/C)]  # Added ESR
        ])
        B = np.array([[D*Vin/L], [-Vin/(R*C)]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([1.0, 1000.0])  # Higher voltage penalty
        R = np.array([[500.0]])  # Much much higher R
        return Q, R
    
    def sample_initial_condition(self):
        return np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-3.0, 3.0)
        ])


class Inverter(LTISystem):
    """
    Three-phase inverter with LC filter (average model, d-q frame)
    State: [id, iq, vd, vq]
    Input: [vd_ref, vq_ref]
    """
    
    def get_default_params(self):
        return {
            'L': 5e-3,    # inductance (H) - increased 5x
            'C': 500e-6,  # capacitance (F) - increased 10x
            'R': 1.0,     # resistance (Ohm) - increased 10x
            'omega': 100  # reduced frequency
        }
    
    def get_matrices(self):
        L, C, R, omega = self.params['L'], self.params['C'], self.params['R'], self.params['omega']
        
        A = np.array([
            [-R/L, omega, -1/L, 0],
            [-omega, -R/L, 0, -1/L],
            [1/C, 0, -0.1/C, omega],  # Added damping
            [0, 1/C, -omega, -0.1/C]   # Added damping
        ])
        B = np.array([
            [1/L, 0],
            [0, 1/L],
            [0, 0],
            [0, 0]
        ])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 10.0, 100.0, 100.0])  # Higher voltage penalties
        R = np.diag([10.0, 10.0])  # Much higher R
        return Q, R
    
    def sample_initial_condition(self):
        return np.array([
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-50.0, 50.0),
            np.random.uniform(-50.0, 50.0)
        ])


class RLCCircuitSeries(LTISystem):
    """
    Series RLC circuit
    State: [capacitor_voltage, inductor_current]
    Input: [voltage_source]
    """
    
    def get_default_params(self):
        return {
            'R': 50.0,    # resistance (Ohm) - increased 5x
            'L': 0.5,     # inductance (H) - increased 5x
            'C': 500e-6   # capacitance (F) - increased 5x
        }
    
    def get_matrices(self):
        R, L, C = self.params['R'], self.params['L'], self.params['C']
        
        A = np.array([
            [0, 1/C],
            [-1/L, -R/L]
        ])
        B = np.array([[0], [1/L]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([100.0, 10.0])  # Higher penalties
        R = np.array([[10.0]])  # Much higher R
        return Q, R
    
    def sample_initial_condition(self):
        return np.array([
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-1.0, 1.0)
        ])


class RLCCircuitParallel(LTISystem):
    """
    Parallel RLC circuit
    State: [inductor_current, capacitor_voltage]
    Input: [current_source]
    """
    
    def get_default_params(self):
        return {
            'R': 20.0,    # resistance (Ohm) - lower for more damping
            'L': 2.0,     # inductance (H) - much larger
            'C': 1000e-6  # capacitance (F) - much larger
        }
    
    def get_matrices(self):
        R, L, C = self.params['R'], self.params['L'], self.params['C']
        
        # Add series resistances for realism and stability
        R_series_L = 2.0  # Series resistance in inductor
        R_series_C = 0.5  # Series resistance in capacitor
        
        A = np.array([
            [-(R/L + R_series_L/L), -1/L],  # Added series resistance
            [1/C, -(1/(R*C) + R_series_C/C)]  # Added series resistance
        ])
        B = np.array([[0], [1/C]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 100.0])  # Higher penalties
        R = np.array([[100.0]])  # Much much higher R
        return Q, R
    
    def sample_initial_condition(self):
        return np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-10.0, 10.0)
        ])

