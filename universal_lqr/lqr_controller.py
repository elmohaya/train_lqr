"""
LQR Controller Implementation with Stability Verification (JAX version)
"""

import jax.numpy as jnp
from jax import jit
import scipy.linalg as sp_linalg  # Still use scipy for ARE solving


def compute_lqr_gain(A, B, Q, R):
    """
    Compute the LQR gain matrix K for the continuous-time system:
        dx/dt = Ax + Bu
    
    The optimal control is u = -Kx
    
    Args:
        A: State matrix (n x n) - can be JAX or numpy array
        B: Input matrix (n x m) - can be JAX or numpy array
        Q: State cost matrix (n x n) - can be JAX or numpy array
        R: Input cost matrix (m x m) - can be JAX or numpy array
    
    Returns:
        K: LQR gain matrix (m x n) - JAX array
        P: Solution to the Riccati equation - JAX array
        success: Boolean indicating if LQR was successfully computed
    """
    import numpy as np
    
    try:
        # Convert to numpy for scipy
        A_np = np.array(A)
        B_np = np.array(B)
        Q_np = np.array(Q)
        R_np = np.array(R)
        
        # Solve continuous-time algebraic Riccati equation
        P = sp_linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)
        
        # Compute LQR gain
        K = sp_linalg.inv(R_np) @ B_np.T @ P
        
        # Convert back to JAX arrays
        K_jax = jnp.array(K)
        P_jax = jnp.array(P)
        
        return K_jax, P_jax, True
    except Exception as e:
        print(f"LQR computation failed: {e}")
        return None, None, False


def verify_stability(A, B, K, tolerance=1e-6):
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
        eigenvalues: All eigenvalues
    """
    import numpy as np
    
    # Closed-loop system matrix
    A_cl = np.array(A) - np.array(B) @ np.array(K)
    
    # Compute eigenvalues
    eigenvalues = sp_linalg.eigvals(A_cl)
    
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
        K: LQR gain matrix - JAX array
        Q: State cost matrix used - JAX array
        R: Input cost matrix used - JAX array
        success: Boolean indicating success
    """
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    
    # Create Q matrix
    if custom_Q is not None:
        Q = custom_Q
    else:
        Q = Q_weight * jnp.eye(n_states)
    
    # Create R matrix
    if custom_R is not None:
        R = custom_R
    else:
        R = R_weight * jnp.eye(n_inputs)
    
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


@jit
def simulate_lqr_step(x, A, B, K, dt, noise_std, key):
    """
    Single step of LQR-controlled system simulation (JIT-compiled).
    
    Args:
        x: Current state
        A: State matrix
        B: Input matrix
        K: LQR gain
        dt: Time step
        noise_std: Process noise standard deviation
        key: JAX random key
    
    Returns:
        x_next: Next state
        u: Control input
        key: Updated random key
    """
    from jax import random
    
    # Compute control
    u = -K @ x
    
    # RK4 integration
    k1 = A @ x + B @ u
    k2 = A @ (x + 0.5 * dt * k1) + B @ u
    k3 = A @ (x + 0.5 * dt * k2) + B @ u
    k4 = A @ (x + dt * k3) + B @ u
    
    # Add process noise
    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=x.shape) * noise_std
    
    # RK4 update
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4) + noise
    
    return x_next, u, key


def simulate_lqr_controlled_system(A, B, K, x0, t_span, dt, process_noise_std=0.0, 
                                   uncertain_A=None, uncertain_B=None,
                                   early_stop=True, settling_threshold=0.01, 
                                   settling_duration=1.0, rng_key=None):
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
        rng_key: JAX random key
    
    Returns:
        t: Time vector
        X: State trajectory (time x n_states)
        U: Control trajectory (time x n_inputs)
    """
    import numpy as np
    from jax import random
    
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    
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
    
    # Set initial condition (convert to numpy for storage)
    X[0] = np.array(x0)
    
    # Early stopping parameters
    settling_steps_required = int(settling_duration / dt)
    settled_counter = 0
    
    # Current state (JAX array for computation)
    x_current = jnp.array(x0)
    
    # Simulate
    actual_steps = n_steps_max
    for i in range(n_steps_max - 1):
        # Simulate one step
        x_next, u, rng_key = simulate_lqr_step(
            x_current, A_sim, B_sim, K, dt, process_noise_std, rng_key
        )
        
        # Store (convert to numpy)
        X[i + 1] = np.array(x_next)
        U[i] = np.array(u)
        
        # Update current state
        x_current = x_next
        
        # Check for early stopping
        if early_stop:
            state_norm = float(jnp.linalg.norm(x_next))
            
            if state_norm < settling_threshold:
                settled_counter += 1
                if settled_counter >= settling_steps_required:
                    actual_steps = i + 2
                    break
            else:
                settled_counter = 0
    
    # Final control
    if actual_steps < n_steps_max:
        U[actual_steps - 1] = np.array(-K @ x_current)
    else:
        U[-1] = np.array(-K @ jnp.array(X[-1]))
    
    # Return only the actually simulated portion
    t = t_max[:actual_steps]
    X = X[:actual_steps]
    U = U[:actual_steps]
    
    return t, X, U

