"""
Configuration file for Universal LQR Transformer Training
All user-adjustable parameters are defined here
"""

import numpy as np

# ============================================================================
# DATA GENERATION PARAMETERS
# ============================================================================

# Enable/disable data saving (if False, will load existing data)
SAVE_NEW_DATA = True

# Number of trajectories to generate per system variant
NUM_TRAJECTORIES_PER_VARIANT = 500

# Time parameters
TIME_HORIZON = 50.0  # seconds (2500 timesteps per trajectory)
DT = 0.02   # sampling time (50 Hz)

# Number of variants per system family (different parameters)
NUM_VARIANTS_PER_SYSTEM = 50

# Parameter uncertainty range (±10% - reduced for numerical stability)
PARAMETER_UNCERTAINTY = 0.10

# Process noise standard deviation (applied to states)
# This will be scaled by the typical state magnitude for each system
PROCESS_NOISE_STD = 0.01

# ============================================================================
# TRANSFORMER PARAMETERS
# ============================================================================

# Sequence length for transformer (number of past timesteps to condition on)
SEQUENCE_LENGTH = 32  # Reduced from 64 for faster training

# Sequence stride (how many timesteps to skip between consecutive sequences)
# stride=1: maximum overlap (~2.16B sequences, too much)
# stride=21: optimized for 100M sequences with 34 systems × 50 variants × 500 trajectories [CURRENT]
# stride=50: ~43M sequences
# stride=72: ~30M sequences
# Formula: num_sequences = num_systems × num_variants × num_trajs × ((T - seq_len) / stride)
# For 100M: 34 × 50 × 500 × ((2500 - 32) / stride) = 100M → stride ≈ 21
SEQUENCE_STRIDE = 21  # Optimized for ~100M input-output pairs

# System dimension parameters (for padding and masking)
MAX_STATE_DIM = 12  # Maximum state dimension across all systems
MAX_INPUT_DIM = 6   # Maximum input dimension across all systems

# Binary encoding dimensions
# We encode n_u and n_x to help transformer understand active dimensions
N_U_ENCODING_BITS = 3  # 3 bits can represent 0-7 (we need 0-6)
N_X_ENCODING_BITS = 4  # 4 bits can represent 0-15 (we need 0-12)

# Total encoding dimension (concatenated to each state vector)
# Encoding: [n_u_binary, n_x_binary]
DIMENSION_ENCODING_SIZE = N_U_ENCODING_BITS + N_X_ENCODING_BITS  # 7 bits total

# Transformer architecture (optimized for ~200k parameters)
TRANSFORMER_CONFIG = {
    'd_model': 64,           # Embedding dimension (reduced for efficiency)
    'n_heads': 4,            # Number of attention heads
    'n_layers': 4,           # Number of transformer blocks
    'd_ff': 256,             # Feed-forward dimension (4x d_model)
    'dropout': 0.1,          # Dropout rate
    'max_seq_len': 128,      # Maximum sequence length (should be >= SEQUENCE_LENGTH)
    
    # Input/Output dimensions
    'input_dim': MAX_STATE_DIM + DIMENSION_ENCODING_SIZE,  # x + [n_u, n_x] encoding
    'output_dim': MAX_INPUT_DIM,  # Fixed output size (will be masked)
}

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

TRAINING_CONFIG = {
    'batch_size': 256,  # CPU-optimized batch size (use 2048 for GPU)
    'learning_rate': 3e-4,  # AdamW learning rate
    'num_epochs': 50,  # More epochs for larger dataset
    'warmup_steps': 2000,
    'gradient_clip': 1.0,
    'validation_split': 0.15,  # 15% validation as requested
    'save_every': 5,  # Save checkpoint every N epochs
    'weight_decay': 1e-4,  # L2 regularization
}

# Data normalization strategy
# Options: 'global' (normalize across all systems), 'per_trajectory', 'standardize'
NORMALIZATION_STRATEGY = 'standardize'  # Standardize to zero mean, unit variance

# ============================================================================
# LQR TUNING PARAMETERS
# ============================================================================

# Default LQR weights (can be overridden per system)
DEFAULT_Q_WEIGHT = 1.0  # State cost weight
DEFAULT_R_WEIGHT = 0.1  # Control cost weight

# LQR stability verification tolerance
LQR_STABILITY_CHECK_TOLERANCE = 1e-6

# ============================================================================
# INITIAL CONDITIONS
# ============================================================================

# Initial condition sampling strategy
# Random initial conditions within reasonable bounds for each system
# We sample from a uniform distribution scaled by system-specific bounds
IC_SAMPLING_STRATEGY = 'uniform'  # Options: 'uniform', 'gaussian'
IC_SCALE_FACTOR = 0.5  # Scale factor for initial condition bounds

# ============================================================================
# OUTPUT PATHS
# ============================================================================

DATA_DIR = './data'
MODEL_DIR = './models'
LOG_DIR = './logs'

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

