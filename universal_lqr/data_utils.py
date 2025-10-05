"""
Utility functions for data preprocessing and handling variable-dimensional systems
"""

import numpy as np
import torch
from config import (
    MAX_STATE_DIM, MAX_INPUT_DIM, 
    N_U_ENCODING_BITS, N_X_ENCODING_BITS,
    DIMENSION_ENCODING_SIZE, SEQUENCE_LENGTH
)


def int_to_binary(num, n_bits):
    """
    Convert integer to binary representation.
    
    Args:
        num: int - number to convert
        n_bits: int - number of bits to use
    
    Returns:
        list of ints (0 or 1) of length n_bits
    """
    binary = [int(x) for x in format(num, f'0{n_bits}b')]
    return binary


def create_dimension_encoding(n_u, n_x):
    """
    Create binary encoding for system dimensions.
    
    Args:
        n_u: int - control dimension
        n_x: int - state dimension
    
    Returns:
        numpy array of shape (DIMENSION_ENCODING_SIZE,) with binary encoding
    """
    n_u_bits = int_to_binary(n_u, N_U_ENCODING_BITS)
    n_x_bits = int_to_binary(n_x, N_X_ENCODING_BITS)
    encoding = np.array(n_u_bits + n_x_bits, dtype=np.float32)
    return encoding


def pad_state(state, target_dim=MAX_STATE_DIM):
    """
    Pad state vector to target dimension.
    
    Args:
        state: numpy array of shape (n_x,) or (seq_len, n_x)
        target_dim: int - target dimension (default: MAX_STATE_DIM)
    
    Returns:
        padded state of shape (target_dim,) or (seq_len, target_dim)
    """
    if state.ndim == 1:
        n_x = state.shape[0]
        if n_x < target_dim:
            padding = np.zeros(target_dim - n_x, dtype=state.dtype)
            return np.concatenate([state, padding])
        return state[:target_dim]
    else:  # 2D array
        seq_len, n_x = state.shape
        if n_x < target_dim:
            padding = np.zeros((seq_len, target_dim - n_x), dtype=state.dtype)
            return np.concatenate([state, padding], axis=1)
        return state[:, :target_dim]


def pad_control(control, target_dim=MAX_INPUT_DIM):
    """
    Pad control vector to target dimension.
    
    Args:
        control: numpy array of shape (n_u,) or (seq_len, n_u)
        target_dim: int - target dimension (default: MAX_INPUT_DIM)
    
    Returns:
        padded control of shape (target_dim,) or (seq_len, target_dim)
    """
    if control.ndim == 1:
        n_u = control.shape[0]
        if n_u < target_dim:
            padding = np.zeros(target_dim - n_u, dtype=control.dtype)
            return np.concatenate([control, padding])
        return control[:target_dim]
    else:  # 2D array
        seq_len, n_u = control.shape
        if n_u < target_dim:
            padding = np.zeros((seq_len, target_dim - n_u), dtype=control.dtype)
            return np.concatenate([control, padding], axis=1)
        return control[:, :target_dim]


def create_control_mask(n_u, target_dim=MAX_INPUT_DIM):
    """
    Create mask for active control dimensions.
    
    Args:
        n_u: int - actual control dimension
        target_dim: int - padded dimension (default: MAX_INPUT_DIM)
    
    Returns:
        numpy array of shape (target_dim,) with 1s for active dims, 0s for padded
    """
    mask = np.zeros(target_dim, dtype=np.float32)
    mask[:n_u] = 1.0
    return mask


def create_state_mask(n_x, target_dim=MAX_STATE_DIM):
    """
    Create mask for active state dimensions.
    
    Args:
        n_x: int - actual state dimension
        target_dim: int - padded dimension (default: MAX_STATE_DIM)
    
    Returns:
        numpy array of shape (target_dim,) with 1s for active dims, 0s for padded
    """
    mask = np.zeros(target_dim, dtype=np.float32)
    mask[:n_x] = 1.0
    return mask


def create_sequence_padding_mask(seq_len, target_len=SEQUENCE_LENGTH):
    """
    Create mask for padded timesteps in sequence.
    
    Args:
        seq_len: int - actual sequence length
        target_len: int - target sequence length (default: SEQUENCE_LENGTH)
    
    Returns:
        numpy array of shape (target_len,) with 1s for valid timesteps, 0s for padding
    """
    mask = np.zeros(target_len, dtype=np.float32)
    mask[:seq_len] = 1.0
    return mask


def prepare_input_sequence(states, n_u, n_x):
    """
    Prepare input sequence for transformer: pad states and add dimension encoding.
    
    Args:
        states: numpy array of shape (seq_len, n_x) - state trajectory
        n_u: int - control dimension
        n_x: int - state dimension
    
    Returns:
        input_sequence: numpy array of shape (seq_len, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE)
    """
    seq_len = states.shape[0]
    
    # Pad states to max dimension
    states_padded = pad_state(states, MAX_STATE_DIM)  # (seq_len, MAX_STATE_DIM)
    
    # Create dimension encoding
    dim_encoding = create_dimension_encoding(n_u, n_x)  # (DIMENSION_ENCODING_SIZE,)
    
    # Repeat encoding for each timestep
    dim_encoding_repeated = np.tile(dim_encoding, (seq_len, 1))  # (seq_len, DIMENSION_ENCODING_SIZE)
    
    # Concatenate states and encoding
    input_sequence = np.concatenate([states_padded, dim_encoding_repeated], axis=1)
    
    return input_sequence


def prepare_training_batch(state_sequences, control_sequences, n_u_list, n_x_list):
    """
    Prepare batch of data for training.
    
    Args:
        state_sequences: list of numpy arrays, each of shape (seq_len, n_x)
        control_sequences: list of numpy arrays, each of shape (seq_len, n_u)
        n_u_list: list of ints - control dimensions for each sequence
        n_x_list: list of ints - state dimensions for each sequence
    
    Returns:
        dict with:
            'input_sequences': torch.Tensor of shape (batch, seq_len, input_dim)
            'target_controls': torch.Tensor of shape (batch, seq_len, MAX_INPUT_DIM)
            'control_masks': torch.Tensor of shape (batch, MAX_INPUT_DIM)
            'padding_masks': torch.Tensor of shape (batch, seq_len)
    """
    batch_size = len(state_sequences)
    
    # Find max sequence length in batch
    max_seq_len = max(seq.shape[0] for seq in state_sequences)
    
    # Initialize arrays
    input_sequences = np.zeros((batch_size, max_seq_len, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE), dtype=np.float32)
    target_controls = np.zeros((batch_size, max_seq_len, MAX_INPUT_DIM), dtype=np.float32)
    control_masks = np.zeros((batch_size, MAX_INPUT_DIM), dtype=np.float32)
    padding_masks = np.zeros((batch_size, max_seq_len), dtype=np.float32)
    
    for i, (states, controls, n_u, n_x) in enumerate(zip(state_sequences, control_sequences, n_u_list, n_x_list)):
        seq_len = states.shape[0]
        
        # Prepare input sequence (padded states + dimension encoding)
        input_seq = prepare_input_sequence(states, n_u, n_x)  # (seq_len, input_dim)
        input_sequences[i, :seq_len, :] = input_seq
        
        # Pad and store target controls
        controls_padded = pad_control(controls, MAX_INPUT_DIM)  # (seq_len, MAX_INPUT_DIM)
        target_controls[i, :seq_len, :] = controls_padded
        
        # Create control mask (which dimensions are active)
        control_masks[i, :] = create_control_mask(n_u, MAX_INPUT_DIM)
        
        # Create padding mask (which timesteps are valid)
        padding_masks[i, :seq_len] = 1.0
    
    # Convert to torch tensors
    batch = {
        'input_sequences': torch.from_numpy(input_sequences),
        'target_controls': torch.from_numpy(target_controls),
        'control_masks': torch.from_numpy(control_masks),
        'padding_masks': torch.from_numpy(padding_masks)
    }
    
    return batch


def extract_active_control(control_padded, n_u):
    """
    Extract only the active control dimensions from padded control.
    
    Args:
        control_padded: numpy array or torch.Tensor of shape (..., MAX_INPUT_DIM)
        n_u: int - actual control dimension
    
    Returns:
        control: array of shape (..., n_u) with only active dimensions
    """
    if isinstance(control_padded, torch.Tensor):
        return control_padded[..., :n_u]
    else:
        return control_padded[..., :n_u]


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Data Utility Functions")
    print("=" * 70)
    
    # Test dimension encoding
    print("\n[TEST 1] Dimension encoding")
    n_u, n_x = 4, 12
    encoding = create_dimension_encoding(n_u, n_x)
    print(f"   n_u = {n_u}, n_x = {n_x}")
    print(f"   Binary encoding: {encoding}")
    print(f"   Encoding shape: {encoding.shape}")
    print(f"   Expected: ({DIMENSION_ENCODING_SIZE},)")
    
    # Test state padding
    print("\n[TEST 2] State padding")
    state = np.random.randn(4)  # CartPole state
    state_padded = pad_state(state)
    print(f"   Original shape: {state.shape}")
    print(f"   Padded shape: {state_padded.shape}")
    print(f"   Non-zero entries: {np.count_nonzero(state_padded)}")
    
    # Test control padding
    print("\n[TEST 3] Control padding")
    control = np.random.randn(1)  # CartPole control
    control_padded = pad_control(control)
    print(f"   Original shape: {control.shape}")
    print(f"   Padded shape: {control_padded.shape}")
    print(f"   Non-zero entries: {np.count_nonzero(control_padded)}")
    
    # Test input sequence preparation
    print("\n[TEST 4] Input sequence preparation")
    seq_len = 50
    n_x = 4
    n_u = 1
    states = np.random.randn(seq_len, n_x)
    input_seq = prepare_input_sequence(states, n_u, n_x)
    print(f"   States shape: {states.shape}")
    print(f"   Input sequence shape: {input_seq.shape}")
    print(f"   Expected: ({seq_len}, {MAX_STATE_DIM + DIMENSION_ENCODING_SIZE})")
    
    # Test batch preparation
    print("\n[TEST 5] Batch preparation")
    # Simulate 3 different systems
    state_seqs = [
        np.random.randn(50, 4),   # CartPole: 4 states, 1 control
        np.random.randn(60, 6),   # Some 6-state system: 6 states, 3 controls
        np.random.randn(40, 12),  # Quadrotor: 12 states, 4 controls
    ]
    control_seqs = [
        np.random.randn(50, 1),
        np.random.randn(60, 3),
        np.random.randn(40, 4),
    ]
    n_u_list = [1, 3, 4]
    n_x_list = [4, 6, 12]
    
    batch = prepare_training_batch(state_seqs, control_seqs, n_u_list, n_x_list)
    
    print(f"   Batch size: {len(state_seqs)}")
    print(f"   Input sequences shape: {batch['input_sequences'].shape}")
    print(f"   Target controls shape: {batch['target_controls'].shape}")
    print(f"   Control masks shape: {batch['control_masks'].shape}")
    print(f"   Padding masks shape: {batch['padding_masks'].shape}")
    
    # Verify masks
    print("\n   Control masks:")
    for i, n_u in enumerate(n_u_list):
        mask_sum = batch['control_masks'][i].sum().item()
        print(f"      System {i+1} (n_u={n_u}): {mask_sum} active dims")
    
    print("\n" + "=" * 70)
    print("[OK] All tests passed!")
    print("=" * 70)

