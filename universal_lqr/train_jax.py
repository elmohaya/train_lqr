"""
Fast Training Script using JAX/Flax and Pre-Processed HDF5 Data

Much faster than PyTorch on multi-GPU systems!
Run data_generation.py first to create training_data.h5
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import h5py
import os
from tqdm import tqdm
from functools import partial
from typing import Any

from config import (
    TRANSFORMER_CONFIG, TRAINING_CONFIG,
    DATA_DIR, MODEL_DIR, RANDOM_SEED,
    MAX_STATE_DIM, MAX_INPUT_DIM, DIMENSION_ENCODING_SIZE, SEQUENCE_LENGTH
)

# Enable GPU memory preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    d_model: int
    n_heads: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask=None, training=True):
        batch_size, seq_len, _ = x.shape
        head_dim = self.d_model // self.n_heads
        
        # Linear projections
        qkv = nn.Dense(3 * self.d_model)(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        
        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention scores
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(head_dim)
        
        # Apply mask
        if mask is not None:
            scores = jnp.where(mask == 0, -1e4, scores)
        
        # Attention weights
        attn_weights = nn.softmax(scores, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout, deterministic=not training)(attn_weights)
        
        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = nn.Dense(self.d_model)(attn_output)
        output = nn.Dropout(rate=self.dropout, deterministic=not training)(output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward."""
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask=None, training=True):
        # Self-attention with residual
        attn_out = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout
        )(x, mask=mask, training=training)
        x = nn.LayerNorm()(x + attn_out)
        
        # Feed-forward with residual
        ff_out = nn.Dense(self.d_ff)(x)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dropout(rate=self.dropout, deterministic=not training)(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        ff_out = nn.Dropout(rate=self.dropout, deterministic=not training)(ff_out)
        x = nn.LayerNorm()(x + ff_out)
        
        return x


class UniversalLQRTransformer(nn.Module):
    """JAX/Flax Universal LQR Transformer."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 128
    max_state_dim: int = 12
    max_control_dim: int = 6
    dimension_encoding_size: int = 7
    
    @nn.compact
    def __call__(self, x, control_mask=None, training=True):
        batch_size, seq_len, input_dim = x.shape
        
        # Input embedding
        x = nn.Dense(self.d_model)(x)
        
        # Positional encoding
        pos_encoding = self.param(
            'pos_encoding',
            nn.initializers.normal(stddev=0.02),
            (1, self.max_seq_len, self.d_model)
        )
        x = x + pos_encoding[:, :seq_len, :]
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        
        # Causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = causal_mask.reshape(1, 1, seq_len, seq_len)
        
        # Transformer blocks
        for _ in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            )(x, mask=causal_mask, training=training)
        
        # Output projection
        output = nn.Dense(self.max_control_dim)(x)
        
        return output


def create_train_state(rng, learning_rate):
    """Create initial training state."""
    model = UniversalLQRTransformer(
        d_model=TRANSFORMER_CONFIG['d_model'],
        n_heads=TRANSFORMER_CONFIG['n_heads'],
        n_layers=TRANSFORMER_CONFIG['n_layers'],
        d_ff=TRANSFORMER_CONFIG['d_ff'],
        dropout=TRANSFORMER_CONFIG['dropout'],
        max_seq_len=TRANSFORMER_CONFIG['max_seq_len'],
        max_state_dim=MAX_STATE_DIM,
        max_control_dim=MAX_INPUT_DIM,
        dimension_encoding_size=DIMENSION_ENCODING_SIZE
    )
    
    # Initialize
    dummy_input = jnp.ones((1, SEQUENCE_LENGTH, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE))
    variables = model.init(rng, dummy_input, training=False)
    
    # Create optimizer with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=TRAINING_CONFIG['warmup_steps'],
        decay_steps=TRAINING_CONFIG['num_epochs'] * 67000,  # Approximate
        end_value=learning_rate * 0.1
    )
    tx = optax.chain(
        optax.clip_by_global_norm(TRAINING_CONFIG['gradient_clip']),
        optax.adam(schedule)
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    ), variables


@partial(jax.jit, static_argnums=(4,))
def train_step(state, batch_inputs, batch_targets, batch_masks, training=True):
    """Single training step (JIT compiled)."""
    
    def loss_fn(params):
        # Forward pass
        outputs = state.apply_fn(
            {'params': params},
            batch_inputs,
            training=training
        )
        
        # Extract last timestep
        outputs_last = outputs[:, -1, :]
        
        # Masked MSE loss
        loss_per_dim = (outputs_last - batch_targets) ** 2
        masked_loss = loss_per_dim * batch_masks
        loss = jnp.sum(masked_loss) / jnp.sum(batch_masks)
        
        return loss
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, loss


@partial(jax.jit, static_argnums=(4,))
def eval_step(state, batch_inputs, batch_targets, batch_masks, training=False):
    """Single evaluation step (JIT compiled)."""
    
    # Forward pass
    outputs = state.apply_fn(
        {'params': state.params},
        batch_inputs,
        training=training
    )
    
    # Extract last timestep
    outputs_last = outputs[:, -1, :]
    
    # Masked MSE loss
    loss_per_dim = (outputs_last - batch_targets) ** 2
    masked_loss = loss_per_dim * batch_masks
    loss = jnp.sum(masked_loss) / jnp.sum(batch_masks)
    
    return loss


def load_batch_from_h5(h5file, indices, start_idx, batch_size):
    """Load a batch from HDF5 file."""
    end_idx = min(start_idx + batch_size, len(indices))
    batch_indices = indices[start_idx:end_idx]
    
    inputs = h5file['input_sequences'][batch_indices]
    controls = h5file['controls'][batch_indices]
    masks = h5file['control_masks'][batch_indices]
    
    return (
        jnp.array(inputs),
        jnp.array(controls),
        jnp.array(masks)
    )


def train_epoch(state, h5file, train_indices, batch_size, rng):
    """Train for one epoch."""
    n_batches = len(train_indices) // batch_size
    epoch_loss = 0.0
    
    # Shuffle indices
    rng, shuffle_rng = random.split(rng)
    perm = random.permutation(shuffle_rng, len(train_indices))
    shuffled_indices = train_indices[perm]
    
    for i in tqdm(range(n_batches), desc="Training", dynamic_ncols=True):
        batch_inputs, batch_targets, batch_masks = load_batch_from_h5(
            h5file, shuffled_indices, i * batch_size, batch_size
        )
        
        state, loss = train_step(state, batch_inputs, batch_targets, batch_masks, training=True)
        epoch_loss += loss
    
    return state, epoch_loss / n_batches, rng


def validate(state, h5file, test_indices, batch_size):
    """Validate the model."""
    n_batches = len(test_indices) // batch_size
    val_loss = 0.0
    
    for i in tqdm(range(n_batches), desc="Validation", dynamic_ncols=True):
        batch_inputs, batch_targets, batch_masks = load_batch_from_h5(
            h5file, test_indices, i * batch_size, batch_size
        )
        
        loss = eval_step(state, batch_inputs, batch_targets, batch_masks, training=False)
        val_loss += loss
    
    return val_loss / n_batches


def main():
    print("="*70)
    print("JAX/Flax Universal LQR Transformer Training")
    print("="*70)
    
    # Check devices
    print(f"\nJAX devices: {jax.devices()}")
    print(f"Number of devices: {jax.device_count()}")
    
    # Load data
    h5_file = os.path.join(DATA_DIR, 'training_data.h5')
    
    if not os.path.exists(h5_file):
        print(f"\nERROR: Training data not found: {h5_file}")
        print("Please run: python data_generation.py")
        return
    
    print(f"\nLoading training data from: {h5_file}")
    with h5py.File(h5_file, 'r') as hf:
        total_sequences = hf['input_sequences'].shape[0]
        print(f"Total sequences: {total_sequences:,}")
    
    # Create train/test split
    all_indices = np.arange(total_sequences)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_indices)
    
    test_size = int(total_sequences * 0.05)
    train_size = total_sequences - test_size
    
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]
    
    print(f"Train size: {train_size:,} (95%)")
    print(f"Test size: {test_size:,} (5%)")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Batches per epoch: {train_size // TRAINING_CONFIG['batch_size']:,}")
    
    # Initialize model
    print("\n=== Creating Model ===")
    rng = random.PRNGKey(RANDOM_SEED)
    rng, init_rng = random.split(rng)
    
    state, variables = create_train_state(init_rng, TRAINING_CONFIG['learning_rate'])
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {param_count:,}")
    
    # Training loop
    print("\n=== Starting Training ===\n")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    # Open HDF5 file for entire training
    with h5py.File(h5_file, 'r') as hf:
        for epoch in range(TRAINING_CONFIG['num_epochs']):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
            print(f"{'='*70}")
            
            # Train
            state, train_loss, rng = train_epoch(
                state, hf, train_indices,
                TRAINING_CONFIG['batch_size'], rng
            )
            train_loss = float(train_loss)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = validate(
                state, hf, test_indices,
                TRAINING_CONFIG['batch_size']
            )
            val_loss = float(val_loss)
            history['val_loss'].append(val_loss)
            
            print(f"\nTrain Loss: {train_loss:.6f} | Test Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save using Flax's checkpointing
                import pickle
                with open(os.path.join(MODEL_DIR, 'best_model_jax.pkl'), 'wb') as f:
                    pickle.dump(state.params, f)
                print(f"[OK] Saved best model (test_loss: {val_loss:.6f})")
            
            # Save checkpoint
            if (epoch + 1) % TRAINING_CONFIG['save_every'] == 0:
                import pickle
                checkpoint = {
                    'epoch': epoch,
                    'params': state.params,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                with open(os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch+1}_jax.pkl'), 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"[OK] Saved checkpoint at epoch {epoch+1}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Best test loss: {best_val_loss:.6f}")
    print(f"Model saved to: {MODEL_DIR}")
    print("="*70)
    
    # Save training history
    import json
    with open(os.path.join(MODEL_DIR, 'training_history_jax.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()

