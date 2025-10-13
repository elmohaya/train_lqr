"""
Configuration optimized for GPU training on Great Lakes
Uses larger batch sizes and full model for best GPU utilization
"""

# Import base config
from config import *

# Override batch size for GPU efficiency
TRAINING_CONFIG['batch_size'] = 2048  # Large batch for GPU (vs 256 for CPU)
TRAINING_CONFIG['num_epochs'] = 50    # Full training

# Optional: Increase learning rate slightly for larger batch
# TRAINING_CONFIG['learning_rate'] = 5e-4  # vs 3e-4 default

print("="*70)
print("=" + " "*18 + "GPU CONFIG LOADED" + " "*33 + "=")
print("="*70)
print(f"Batch size: {TRAINING_CONFIG['batch_size']} (optimized for GPU)")
print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
print(f"Model size: ~{208_000:,} parameters")
print(f"Expected training time on V100: ~3-4 hours")
print(f"Expected training time on A100: ~1-2 hours")
print("="*70)

