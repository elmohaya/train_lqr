"""
JAX/Flax Transformer for Universal LQR Control
Optimized for JIT compilation and multi-GPU training
~200k parameters
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Optional
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask=None, training=True):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or None
            training: bool
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        d_k = self.d_model // self.n_heads
        
        # Linear projections
        W_q = nn.Dense(self.d_model, use_bias=False, name='query')
        W_k = nn.Dense(self.d_model, use_bias=False, name='key')
        W_v = nn.Dense(self.d_model, use_bias=False, name='value')
        W_o = nn.Dense(self.d_model, use_bias=False, name='output')
        
        Q = W_q(x).reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        K = W_k(x).reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        V = W_v(x).reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)
        
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)
        
        attn = nn.softmax(scores, axis=-1)
        attn = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn)
        
        context = jnp.matmul(attn, V)
        
        # Concatenate heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = W_o(context)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    d_model: int
    d_ff: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=True):
        """
        Args:
            x: (batch, seq_len, d_model)
            training: bool
        Returns:
            output: (batch, seq_len, d_model)
        """
        x = nn.Dense(self.d_ff, name='fc1')(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.d_model, name='fc2')(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask=None, training=True):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or None
            training: bool
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_out = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
        )(x, mask=mask, training=training)
        attn_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn_out)
        x = nn.LayerNorm()(x + attn_out)
        
        # Feed-forward with residual connection and layer norm
        ff_out = FeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate
        )(x, training=training)
        ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_out)
        x = nn.LayerNorm()(x + ff_out)
        
        return x


class UniversalLQRTransformer(nn.Module):
    """
    Universal LQR Transformer (JAX/Flax version)
    
    Predicts control action u(t) given state history x(0:t).
    Handles variable-dimensional systems via padding and masking.
    ~200k parameters (d_model=64, n_heads=4, n_layers=4, d_ff=256)
    """
    input_dim: int  # MAX_STATE_DIM + DIMENSION_ENCODING_SIZE
    output_dim: int  # MAX_INPUT_DIM
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout_rate: float = 0.1
    max_seq_len: int = 128
    
    @nn.compact
    def __call__(self, x, training=True):
        """
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            training: bool for dropout
        Returns:
            control: Predicted controls (batch, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = nn.Dense(self.d_model, name='input_proj')(x)
        
        # Add positional encoding
        pos_encoding = self.param(
            'pos_encoding',
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, self.d_model)
        )
        x = x + pos_encoding[:seq_len]
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Causal mask for autoregressive prediction
        # This ensures position i can only attend to positions <= i
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = causal_mask.reshape(1, 1, seq_len, seq_len)  # (1, 1, seq_len, seq_len)
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))
        
        # Transformer blocks
        for i in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f'block_{i}'
            )(x, mask=causal_mask, training=training)
        
        # Take the last timestep's representation
        x = x[:, -1, :]  # (batch, d_model)
        
        # Output projection to control dimension
        control = nn.Dense(self.output_dim, name='output_proj')(x)
        
        return control


def create_model(config: dict) -> UniversalLQRTransformer:
    """
    Create model from config dictionary.
    
    Args:
        config: Configuration dictionary with model hyperparameters
    Returns:
        model: UniversalLQRTransformer instance
    """
    return UniversalLQRTransformer(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout_rate=config['dropout'],
        max_seq_len=config['max_seq_len']
    )


def count_parameters(params):
    """Count total number of parameters in model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


if __name__ == '__main__':
    """Test model creation and parameter count"""
    from config import TRANSFORMER_CONFIG, SEQUENCE_LENGTH
    
    print("="*70)
    print("Testing JAX/Flax Transformer Model")
    print("="*70)
    
    # Create model
    model = create_model(TRANSFORMER_CONFIG)
    
    # Initialize with dummy data
    batch_size = 4
    seq_len = SEQUENCE_LENGTH
    input_dim = TRANSFORMER_CONFIG['input_dim']
    
    dummy_input = jnp.ones((batch_size, seq_len, input_dim))
    
    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_input, training=False)
    
    # Count parameters
    n_params = count_parameters(params)
    
    print(f"\nModel Configuration:")
    print(f"  d_model: {TRANSFORMER_CONFIG['d_model']}")
    print(f"  n_heads: {TRANSFORMER_CONFIG['n_heads']}")
    print(f"  n_layers: {TRANSFORMER_CONFIG['n_layers']}")
    print(f"  d_ff: {TRANSFORMER_CONFIG['d_ff']}")
    print(f"  input_dim: {TRANSFORMER_CONFIG['input_dim']}")
    print(f"  output_dim: {TRANSFORMER_CONFIG['output_dim']}")
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Size (MB): {n_params * 4 / (1024**2):.2f}")  # Assuming float32
    
    # Test forward pass
    output = model.apply(params, dummy_input, training=False)
    print(f"\nForward Pass Test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, {TRANSFORMER_CONFIG['output_dim']})")
    
    # Test training mode
    output_train = model.apply(params, dummy_input, training=True, rngs={'dropout': rng})
    print(f"  Training mode output shape: {output_train.shape}")
    
    print("\n" + "="*70)
    print(f"[OK] Model created successfully with {n_params:,} parameters!")
    print("="*70)

