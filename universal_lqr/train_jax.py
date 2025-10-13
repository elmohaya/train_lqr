"""
JAX Training Script for Universal LQR Transformer
JIT-compiled for maximum GPU efficiency with multi-GPU support
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from jax import tree
import optax
from flax.training import train_state
from flax import jax_utils
import numpy as np
import os
import time
from tqdm import tqdm
import pickle
from typing import Dict, Any

# For fast CPU training (12-20 hours), use config_fast_cpu
# For full model (15+ days), use config
from config_fast_cpu import (
    DATA_DIR, MODEL_DIR, LOG_DIR, TRANSFORMER_CONFIG, TRAINING_CONFIG,
    SEQUENCE_LENGTH, MAX_STATE_DIM, MAX_INPUT_DIM, DIMENSION_ENCODING_SIZE
)
from transformer_model_jax import create_model, count_parameters
from data_loader import create_jax_dataloaders

# Create directories if they don't exist
import os as _os
_os.makedirs(DATA_DIR, exist_ok=True)
_os.makedirs(MODEL_DIR, exist_ok=True)
_os.makedirs(LOG_DIR, exist_ok=True)


def masked_mse_loss(predictions, targets, masks):
    """
    Masked MSE loss - only compute loss on active control dimensions.
    
    Args:
        predictions: (batch, output_dim)
        targets: (batch, output_dim)
        masks: (batch, output_dim) - 1 for active dims, 0 for padded
    Returns:
        loss: scalar
    """
    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * masks
    
    # Normalize by number of active elements
    loss = jnp.sum(masked_error) / jnp.maximum(jnp.sum(masks), 1.0)
    
    return loss


def create_train_state(rng, config, learning_rate):
    """
    Create initial training state.
    
    Args:
        rng: Random key
        config: Model config
        learning_rate: Learning rate
    Returns:
        state: Training state
    """
    model = create_model(config)
    
    # Initialize with dummy data
    dummy_input = jnp.ones((1, SEQUENCE_LENGTH, config['input_dim']))
    params = model.init(rng, dummy_input, training=False)
    
    # Count parameters
    n_params = count_parameters(params)
    print(f"Model parameters: {n_params:,}")
    
    # Create optimizer with warmup and cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=TRAINING_CONFIG['warmup_steps'],
        decay_steps=TRAINING_CONFIG['num_epochs'] * 1000,  # Approximate
        end_value=learning_rate * 0.1
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(TRAINING_CONFIG['gradient_clip']),
        optax.adamw(learning_rate=schedule, weight_decay=TRAINING_CONFIG['weight_decay'])
    )
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state


@jit
def train_step(state, batch, rng):
    """
    Single training step (JIT compiled).
    
    Args:
        state: Training state
        batch: Batch dict with input_sequences, controls, control_masks
        rng: Random key for dropout
    Returns:
        state: Updated state
        loss: Scalar loss
    """
    def loss_fn(params):
        predictions = state.apply_fn(
            params, 
            batch['input_sequences'],
            training=True,
            rngs={'dropout': rng}
        )
        loss = masked_mse_loss(predictions, batch['controls'], batch['control_masks'])
        return loss
    
    loss, grads = value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss


@jit
def eval_step(state, batch):
    """
    Single evaluation step (JIT compiled).
    
    Args:
        state: Training state
        batch: Batch dict
    Returns:
        loss: Scalar loss
    """
    predictions = state.apply_fn(
        state.params,
        batch['input_sequences'],
        training=False
    )
    loss = masked_mse_loss(predictions, batch['controls'], batch['control_masks'])
    return loss


def train_epoch(state, train_loader, rng, epoch, device):
    """
    Train for one epoch (memory-efficient with GPU transfer).
    
    Args:
        state: Training state
        train_loader: Training data loader
        rng: Random key
        epoch: Current epoch number
        device: JAX device for GPU transfer
    Returns:
        state: Updated state
        avg_loss: Average training loss
    """
    total_loss = 0.0
    n_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch_idx, batch_np in enumerate(pbar):
        # Transfer to device efficiently (only the data we need)
        batch = tree.map(lambda x: jax.device_put(jnp.array(x), device), batch_np)
        
        # Generate new RNG for this batch
        rng, step_rng = jax.random.split(rng)
        
        # Training step (JIT-compiled, runs on GPU)
        state, loss = train_step(state, batch, step_rng)
        
        # Block until computation is done and get loss value
        loss_value = float(loss)
        total_loss += loss_value
        
        # Reset tqdm timer after JIT compilation (first 2 batches)
        if batch_idx == 2:
            pbar.start_t = time.time()  # Reset timer after compilation
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_value:.6f}', 'avg': f'{total_loss/(batch_idx+1):.6f}'})
    
    avg_loss = total_loss / n_batches
    return state, avg_loss, rng


def evaluate(state, val_loader, epoch, device):
    """
    Evaluate on validation set (memory-efficient with GPU transfer).
    
    Args:
        state: Training state
        val_loader: Validation data loader
        epoch: Current epoch number
        device: JAX device for GPU transfer
    Returns:
        avg_loss: Average validation loss
    """
    total_loss = 0.0
    n_batches = len(val_loader)
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
    
    for batch_idx, batch_np in enumerate(pbar):
        # Transfer to device efficiently
        batch = tree.map(lambda x: jax.device_put(jnp.array(x), device), batch_np)
        
        # Evaluation step (JIT-compiled, runs on GPU)
        loss = eval_step(state, batch)
        
        # Block until computation is done and get loss value
        loss_value = float(loss)
        total_loss += loss_value
        
        pbar.set_postfix({'loss': f'{loss_value:.6f}', 'avg': f'{total_loss/(batch_idx+1):.6f}'})
    
    avg_loss = total_loss / n_batches
    return avg_loss


def save_checkpoint(state, epoch, val_loss, save_dir, filename='checkpoint.pkl'):
    """Save training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'params': state.params,
        'opt_state': state.opt_state,
        'epoch': epoch,
        'val_loss': val_loss
    }
    
    path = os.path.join(save_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"  Checkpoint saved: {path}")


def main():
    """Main training loop (memory-efficient GPU training)."""
    
    training_start_time = time.time()
    
    print("="*80)
    print(" JAX Universal LQR Transformer Training ".center(80, "="))
    print("="*80)
    
    # Setup
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Check for GPUs and get device
    devices = jax.devices()
    device = devices[0]  # Use first device (GPU if available)
    n_devices = len(devices)
    
    print(f"\n{'='*80}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*80}")
    print(f"Available devices: {n_devices}")
    print(f"Device type: {device.device_kind}")
    print(f"Device: {device}")
    print(f"Backend: {jax.default_backend()}")
    
    if device.device_kind == 'METAL':
        print(f"✓ Using Apple Metal GPU acceleration on M4!")
    elif device.device_kind == 'gpu':
        print(f"✓ Using CUDA GPU acceleration!")
    else:
        print(f"⚠ Using CPU (slower training expected)")
    print()
    
    # Load data
    print("\nLoading data...")
    h5_path = os.path.join(DATA_DIR, 'training_data_jax.h5')
    
    if not os.path.exists(h5_path):
        print(f"Error: Data file not found: {h5_path}")
        print("Please run data_generation.py first.")
        return
    
    train_loader, val_loader = create_jax_dataloaders(
        h5_path,
        batch_size=TRAINING_CONFIG['batch_size'],
        validation_split=TRAINING_CONFIG['validation_split']
    )
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Sequences per epoch: {len(train_loader) * TRAINING_CONFIG['batch_size']:,}")
    
    # Save normalization stats
    stats = train_loader.get_normalization_stats()
    stats_path = os.path.join(MODEL_DIR, 'normalization_stats_jax.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"  Normalization stats saved: {stats_path}")
    
    # Create model
    print("\nInitializing model...")
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    state = create_train_state(
        init_rng,
        TRANSFORMER_CONFIG,
        TRAINING_CONFIG['learning_rate']
    )
    
    # Training info
    print("\nTraining Configuration:")
    print(f"  Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Warmup steps: {TRAINING_CONFIG['warmup_steps']}")
    print(f"  Weight decay: {TRAINING_CONFIG['weight_decay']}")
    print(f"  Gradient clip: {TRAINING_CONFIG['gradient_clip']}")
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING (Memory-efficient streaming from HDF5)")
    print("="*80)
    print(f"Note: Data is streamed in batches, never loading full dataset into RAM")
    print()
    
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_times': []
    }
    
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        epoch_start = time.time()
        
        # Train (with GPU device)
        state, train_loss, rng = train_epoch(state, train_loader, rng, epoch, device)
        
        # Evaluate (with GPU device)
        val_loss = evaluate(state, val_loader, epoch, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['epoch_times'].append(epoch_time)
        
        epoch_mins = epoch_time / 60.0
        elapsed_mins = (time.time() - training_start_time) / 60.0
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']} Summary")
        print(f"{'='*80}")
        print(f"  Train Loss:     {train_loss:.6f}")
        print(f"  Val Loss:       {val_loss:.6f}")
        print(f"  Epoch Time:     {epoch_mins:.2f} min ({epoch_time:.1f}s)")
        print(f"  Elapsed Time:   {elapsed_mins:.2f} min")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(state, epoch, val_loss, MODEL_DIR, 'best_model_jax.pkl')
            print(f"  ✓ New best model! Val loss: {val_loss:.6f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % TRAINING_CONFIG['save_every'] == 0:
            save_checkpoint(state, epoch, val_loss, MODEL_DIR, f'checkpoint_epoch_{epoch+1}_jax.pkl')
        
        print("-"*70)
    
    # Final save
    save_checkpoint(state, epoch, val_loss, MODEL_DIR, 'final_model_jax.pkl')
    
    # Save training history
    history_path = os.path.join(LOG_DIR, 'training_history_jax.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(training_history, f)
    
    # Calculate final statistics
    total_time_sec = time.time() - training_start_time
    total_time_min = total_time_sec / 60.0
    total_time_hr = total_time_sec / 3600.0
    avg_epoch_time_min = np.mean(training_history['epoch_times']) / 60.0
    
    print("\n" + "="*80)
    print(" TRAINING COMPLETE! ".center(80, "="))
    print("="*80)
    
    print(f"\n{'FINAL RESULTS':^80}")
    print(f"{'='*80}")
    print(f"  Best Validation Loss:        {best_val_loss:.6f}")
    print(f"  Final Training Loss:         {training_history['train_loss'][-1]:.6f}")
    print(f"  Final Validation Loss:       {training_history['val_loss'][-1]:.6f}")
    
    print(f"\n{'TIMING STATISTICS':^80}")
    print(f"{'='*80}")
    print(f"  Total Training Time:         {total_time_min:.2f} minutes ({total_time_hr:.2f} hours)")
    print(f"  Average Epoch Time:          {avg_epoch_time_min:.2f} minutes")
    print(f"  Total Epochs:                {TRAINING_CONFIG['num_epochs']}")
    print(f"  Device:                      {device.device_kind}")
    
    print(f"\n{'OUTPUT FILES':^80}")
    print(f"{'='*80}")
    print(f"  Best Model:                  {os.path.join(MODEL_DIR, 'best_model_jax.pkl')}")
    print(f"  Final Model:                 {os.path.join(MODEL_DIR, 'final_model_jax.pkl')}")
    print(f"  Training History:            {history_path}")
    print(f"  Normalization Stats:         {os.path.join(MODEL_DIR, 'normalization_stats_jax.pkl')}")
    
    print(f"\n{'='*80}")
    print(f" ✓ Training completed successfully! ".center(80, "="))
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

