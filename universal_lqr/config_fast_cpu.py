"""
FAST CPU Configuration
Optimized for CPU training - should complete in ~12-24 hours
"""

from config import *

# Override with CPU-optimized settings
TRANSFORMER_CONFIG = {
    'd_model': 32,           # REDUCED from 64 (fewer parameters)
    'n_heads': 2,            # REDUCED from 4
    'n_layers': 2,           # REDUCED from 4 (much faster!)
    'd_ff': 128,             # REDUCED from 256
    'dropout': 0.1,
    'max_seq_len': 128,
    'input_dim': MAX_STATE_DIM + DIMENSION_ENCODING_SIZE,
    'output_dim': MAX_INPUT_DIM,
}

TRAINING_CONFIG = {
    'batch_size': 512,       # Larger batch for better vectorization on CPU
    'learning_rate': 3e-4,
    'num_epochs': 20,        # REDUCED from 50 (faster training)
    'warmup_steps': 1000,    # REDUCED from 2000
    'gradient_clip': 1.0,
    'validation_split': 0.15,
    'save_every': 2,         # Save more frequently
    'weight_decay': 1e-4,
}

print("="*70)
print(" FAST CPU CONFIG LOADED ".center(70, "="))
print("="*70)
print(f"Model size: ~40k parameters (vs 200k in normal config)")
print(f"Layers: 2 (vs 4)")
print(f"d_model: 32 (vs 64)")
print(f"Epochs: 20 (vs 50)")
print(f"Expected training time: ~12-20 hours")
print("="*70)

