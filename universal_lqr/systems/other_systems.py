"""
Other LTI Systems (biological, chemical, and mathematical systems)
"""

import numpy as np
from .base_system import LTISystem


class DoubleIntegrator(LTISystem):
    """
    Double integrator system (fundamental control theory system)
    State: [position, velocity]
    Input: [acceleration]
    """
    
    def get_default_params(self):
        return {
            'm': 1.0  # mass (can be varied)
        }
    
    def get_matrices(self):
        m = self.params['m']
        A = np.array([
            [0, 1],
            [0, 0]
        ])
        B = np.array([[0], [1/m]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 1.0])
        R = np.array([[0.1]])
        return Q, R
    
    def sample_initial_condition(self):
        # Double integrator: larger workspace ±5m, higher speeds ±3 m/s
        return np.array([
            np.random.uniform(-5.0, 5.0),   # position (m)
            np.random.uniform(-3.0, 3.0)    # velocity (m/s)
        ])


class LotkaVolterra(LTISystem):
    """
    Lotka-Volterra predator-prey model (linearized near equilibrium)
    State: [prey_deviation, predator_deviation]
    Input: [prey_control, predator_control]
    """
    
    def get_default_params(self):
        return {
            'alpha': 1.0,   # prey growth rate
            'beta': 0.1,    # predation rate
            'gamma': 1.5,   # predator death rate
            'delta': 0.075  # predator growth from prey
        }
    
    def get_matrices(self):
        alpha, beta, gamma, delta = (self.params['alpha'], self.params['beta'],
                                      self.params['gamma'], self.params['delta'])
        
        # Equilibrium point
        x_eq = gamma / delta
        y_eq = alpha / beta
        
        # Jacobian at equilibrium
        A = np.array([
            [0, -beta*x_eq],
            [delta*y_eq, 0]
        ])
        B = np.eye(2)  # Direct control on both populations
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 10.0])
        R = np.diag([1.0, 1.0])
        return Q, R
    
    def sample_initial_condition(self):
        # Lotka-Volterra: population variations ±50% around equilibrium
        # Equilibrium is around [prey=10, predator=5]
        return np.array([
            np.random.uniform(-7.0, 7.0),   # prey deviation from equilibrium
            np.random.uniform(-4.0, 4.0)    # predator deviation from equilibrium
        ])


class ChemicalReactor(LTISystem):
    """
    Continuous Stirred Tank Reactor (CSTR) - linearized
    State: [concentration, temperature]
    Input: [coolant_flow_rate]
    """
    
    def get_default_params(self):
        return {
            'V': 1.0,       # reactor volume (m^3)
            'rho': 1000,    # density (kg/m^3)
            'Cp': 4180,     # heat capacity (J/(kg·K))
            'H': -2e5,      # heat of reaction (J/mol)
            'k0': 7.2e10,   # pre-exponential factor
            'E': 8750,      # activation energy (K)
            'UA': 5e4,      # heat transfer coefficient (J/(s·K))
            'F': 0.1,       # flow rate (m^3/s)
            'Ca0': 10.0,    # inlet concentration (mol/m^3)
            'T0': 350.0,    # inlet temperature (K)
            'Tc': 300.0     # coolant temperature (K)
        }
    
    def get_matrices(self):
        p = self.params
        
        # Steady state (equilibrium point)
        T_ss = 370.0  # steady state temperature - reduced for more stability
        Ca_ss = 6.0   # steady state concentration - increased
        
        # Reaction rate constant at steady state
        k_ss = p['k0'] * np.exp(-p['E'] / T_ss)
        
        # Linearized dynamics with added damping
        A = np.array([
            [-(p['F']/p['V'] + k_ss + 0.1), -p['E']*k_ss*Ca_ss/T_ss**2],  # Added damping
            [-p['H']*k_ss/(p['rho']*p['Cp']), 
             -(p['F']/p['V'] + p['UA']/(p['V']*p['rho']*p['Cp']) + 
               p['H']*p['E']*k_ss*Ca_ss/(p['rho']*p['Cp']*T_ss**2) + 0.05)]  # Added damping
        ])
        
        B = np.array([
            [0],
            [p['UA']/(p['V']*p['rho']*p['Cp'])]
        ])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([200.0, 10.0])  # Much higher penalties for faster convergence
        R = np.array([[1.0]])  # Higher R to prevent aggressive control
        return Q, R
    
    def sample_initial_condition(self):
        # Chemical reactor: concentration 0.5-3.5 mol/L, temperature 250-550K
        # Operating point: C_ss≈2.0 mol/L, T_ss≈400K
        return np.array([
            np.random.uniform(-1.5, 1.5),     # concentration deviation (mol/L) → 0.5-3.5
            np.random.uniform(-150.0, 150.0)  # temperature deviation (K) → 250-550K
        ])
    
    def get_typical_state_magnitude(self):
        return np.array([2.0, 10.0])

