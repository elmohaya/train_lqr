"""
Base class for LTI systems (JAX version)
Matrices use JAX arrays for computation, but IC sampling uses numpy (simpler, not performance-critical)
"""

import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod


class LTISystem(ABC):
    """
    Base class for Linear Time-Invariant systems (JAX implementation).
    
    All systems should inherit from this class and implement:
    - get_matrices(): Returns A, B matrices (JAX arrays)
    - get_default_lqr_weights(): Returns Q, R weights (JAX arrays)
    - sample_initial_condition(): Returns random initial state (numpy array, converted to JAX when needed)
    """
    
    def __init__(self, params=None):
        """
        Initialize system with parameters.
        
        Args:
            params: Dictionary of system parameters
        """
        self.params = params if params is not None else self.get_default_params()
        self.A, self.B = self.get_matrices()
        self.n_states = self.A.shape[0]
        self.n_inputs = self.B.shape[1]
        self.name = self.__class__.__name__
    
    @abstractmethod
    def get_default_params(self):
        """Return default parameter dictionary"""
        pass
    
    @abstractmethod
    def get_matrices(self):
        """
        Return the state-space matrices A and B.
        
        Returns:
            A: State matrix (n x n) - JAX array
            B: Input matrix (n x m) - JAX array
        """
        pass
    
    @abstractmethod
    def get_default_lqr_weights(self):
        """
        Return default LQR weight matrices Q and R.
        
        Returns:
            Q: State cost matrix (n x n) - JAX array
            R: Input cost matrix (m x m) - JAX array
        """
        pass
    
    @abstractmethod
    def sample_initial_condition(self):
        """
        Sample a random initial condition.
        
        Returns:
            x0: Initial state vector (n,) - numpy array
        """
        pass
    
    def generate_variant_params(self, uncertainty_range=0.30):
        """
        Generate variant parameters by perturbing the nominal parameters.
        
        Args:
            uncertainty_range: Relative uncertainty range (e.g., 0.30 for ±30%)
        
        Returns:
            variant_params: Dictionary of perturbed parameters
        """
        variant_params = {}
        for key, value in self.params.items():
            if isinstance(value, (int, float)):
                # Add random perturbation
                perturbation = np.random.uniform(-uncertainty_range, uncertainty_range)
                variant_params[key] = value * (1 + perturbation)
            else:
                variant_params[key] = value
        return variant_params
    
    def get_typical_state_magnitude(self):
        """
        Return typical magnitude of states for noise scaling.
        This should be overridden by specific systems if needed.
        
        Returns:
            magnitude: Array of typical magnitudes for each state - JAX array
        """
        # Default: use unit magnitude, can be overridden
        return jnp.ones(self.n_states)
    
    def __str__(self):
        return f"{self.name}(n_states={self.n_states}, n_inputs={self.n_inputs})"

