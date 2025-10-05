"""
LQR Controller Implementation with Stability Verification
"""

import numpy as np
from scipy import linalg
from config import LQR_STABILITY_CHECK_TOLERANCE


def compute_lqr_gain(A, B, Q, R):
    """
    Compute the LQR gain matrix K for the continuous-time system:
        dx/dt = Ax + Bu
    
    The optimal control is u = -Kx
    
    Args:
        A: State matrix (n x n)
        B: Input matrix (n x m)
        Q: State cost matrix (n x n)
        R: Input cost matrix (m x m)
    
    Returns:
        K: LQR gain matrix (m x n)
        P: Solution to the Riccati equation
        success: Boolean indicating if LQR was successfully computed
    """
    try:
        # Solve continuous-time algebraic Riccati equation
        P = linalg.solve_continuous_are(A, B, Q, R)
        
        # Compute LQR gain
        K = linalg.inv(R) @ B.T @ P
        
        return K, P, True
    except Exception as e:
        print(f"LQR computation failed: {e}")
        return None, None, False


def verify_stability(A, B, K, tolerance=LQR_STABILITY_CHECK_TOLERANCE):
    """
    Verify that the closed-loop system is stable.
    
    The closed-loop system is: dx/dt = (A - BK)x
    System is stable if all eigenvalues have negative real parts.
    
    Args:
        A: State matrix (n x n)
        B: Input matrix (n x m)
        K: LQR gain matrix (m x n)
        tolerance: Tolerance for stability check
    
    Returns:
        is_stable: Boolean
        max_real_eigenvalue: Maximum real part of eigenvalues
    """
    # Closed-loop system matrix
    A_cl = A - B @ K
    
    # Compute eigenvalues
    eigenvalues = linalg.eigvals(A_cl)
    
    # Check if all eigenvalues have negative real parts
    max_real_part = np.max(np.real(eigenvalues))
    is_stable = max_real_part < -tolerance
    
    return is_stable, max_real_part, eigenvalues


def design_lqr(A, B, Q_weight=1.0, R_weight=0.1, custom_Q=None, custom_R=None):
    """
    Design LQR controller with automatic Q and R matrix generation.
    
    Args:
        A: State matrix
        B: Input matrix
        Q_weight: Weight for state cost (if custom_Q not provided)
        R_weight: Weight for input cost (if custom_R not provided)
        custom_Q: Custom Q matrix (optional)
        custom_R: Custom R matrix (optional)
    
    Returns:
        K: LQR gain matrix
        Q: State cost matrix used
        R: Input cost matrix used
        success: Boolean indicating success
    """
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    
    # Create Q matrix
    if custom_Q is not None:
        Q = custom_Q
    else:
        Q = Q_weight * np.eye(n_states)
    
    # Create R matrix
    if custom_R is not None:
        R = custom_R
    else:
        R = R_weight * np.eye(n_inputs)
    
    # Compute LQR gain
    K, P, success = compute_lqr_gain(A, B, Q, R)
    
    if not success:
        return None, Q, R, False
    
    # Verify stability
    is_stable, max_eig, eigenvalues = verify_stability(A, B, K)
    
    if not is_stable:
        print(f"Warning: LQR controller is not stable! Max eigenvalue real part: {max_eig}")
        print(f"Eigenvalues: {eigenvalues}")
        return K, Q, R, False
    
    return K, Q, R, True


def simulate_lqr_controlled_system(A, B, K, x0, t_span, dt, process_noise_std=0.0, 
                                   uncertain_A=None, uncertain_B=None,
                                   early_stop=True, settling_threshold=0.01, 
                                   settling_duration=1.0):
    """
    Simulate the LQR-controlled system with optional uncertainties and noise.
    
    Args:
        A: Nominal state matrix
        B: Nominal input matrix
        K: LQR gain matrix (computed for nominal system)
        x0: Initial state
        t_span: Total simulation time (maximum)
        dt: Time step
        process_noise_std: Standard deviation of process noise
        uncertain_A: Uncertain state matrix (if None, uses nominal A)
        uncertain_B: Uncertain input matrix (if None, uses nominal B)
        early_stop: Enable early stopping when system settles
        settling_threshold: State norm threshold for settling
        settling_duration: Time duration system must stay settled before stopping
    
    Returns:
        t: Time vector
        X: State trajectory (time x n_states)
        U: Control trajectory (time x n_inputs)
    """
    # Use uncertain matrices if provided, otherwise use nominal
    A_sim = uncertain_A if uncertain_A is not None else A
    B_sim = uncertain_B if uncertain_B is not None else B
    
    # Time vector
    t_max = np.arange(0, t_span, dt)
    n_steps_max = len(t_max)
    
    # Initialize trajectories
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    X = np.zeros((n_steps_max, n_states))
    U = np.zeros((n_steps_max, n_inputs))
    
    # Set initial condition
    X[0] = x0
    
    # Early stopping parameters
    settling_steps_required = int(settling_duration / dt)
    settled_counter = 0
    
    # Simulate using RK4 integration (more stable than Euler for stiff systems)
    actual_steps = n_steps_max
    for i in range(n_steps_max - 1):
        # Compute control (LQR uses current state)
        u = -K @ X[i]
        U[i] = u
        
        # RK4 integration (4th order Runge-Kutta)
        # More accurate and stable than Euler, especially for stiff systems
        k1 = A_sim @ X[i] + B_sim @ u
        k2 = A_sim @ (X[i] + 0.5 * dt * k1) + B_sim @ u
        k3 = A_sim @ (X[i] + 0.5 * dt * k2) + B_sim @ u
        k4 = A_sim @ (X[i] + dt * k3) + B_sim @ u
        
        # Add process noise to state
        noise = np.random.randn(n_states) * process_noise_std
        
        # RK4 update
        X[i + 1] = X[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4) + noise
        
        # Check for early stopping
        if early_stop:
            state_norm = np.linalg.norm(X[i + 1])
            
            if state_norm < settling_threshold:
                settled_counter += 1
                # If settled for required duration, stop simulation
                if settled_counter >= settling_steps_required:
                    actual_steps = i + 2  # Include current step
                    break
            else:
                settled_counter = 0  # Reset counter if threshold exceeded
    
    # Final control
    if actual_steps < n_steps_max:
        U[actual_steps - 1] = -K @ X[actual_steps - 1]
    else:
        U[-1] = -K @ X[-1]
    
    # Return only the actually simulated portion
    t = t_max[:actual_steps]
    X = X[:actual_steps]
    U = U[:actual_steps]
    
    return t, X, U

