"""
GPT-style Transformer for Universal LQR Control
Task: Given state sequence x(0:t), predict control u(t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or None
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Single transformer block with multi-head attention and feed-forward.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class UniversalLQRTransformer(nn.Module):
    """
    GPT-style Transformer for Universal LQR Control.
    
    Given a sequence of states x(0:t) with dimension encoding, predict the control u(t).
    The model is system-agnostic and should generalize to unseen LTI systems.
    
    Input format: [x(t), n_u_binary, n_x_binary] at each timestep
    Output format: u(t) with fixed dimension (padded/masked)
    """
    
    def __init__(self, max_state_dim, max_control_dim, dimension_encoding_size=7,
                 d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1, max_seq_len=128):
        """
        Args:
            max_state_dim: Maximum state dimension across all systems
            max_control_dim: Maximum control dimension across all systems
            dimension_encoding_size: Size of binary dimension encoding (n_u + n_x bits)
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.max_state_dim = max_state_dim
        self.max_control_dim = max_control_dim
        self.dimension_encoding_size = dimension_encoding_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input dimension = padded state + dimension encoding
        self.input_dim = max_state_dim + dimension_encoding_size
        
        # Input embedding: project (states + encoding) to model dimension
        self.state_embedding = nn.Linear(self.input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head: predict control
        self.control_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max_control_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len, device):
        """
        Create causal mask for autoregressive prediction.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        
        Returns:
            mask: (seq_len, seq_len) causal mask
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0)  # (1, seq_len, seq_len)
    
    def forward(self, input_sequence, padding_mask=None, control_mask=None):
        """
        Forward pass.
        
        Args:
            input_sequence: (batch, seq_len, input_dim) - padded state sequences + dim encoding
                            input_dim = max_state_dim + dimension_encoding_size
            padding_mask: (batch, seq_len) - mask for padded timesteps (1 = valid, 0 = padding)
            control_mask: (batch, max_control_dim) - mask for active control dims (1 = active, 0 = inactive)
        
        Returns:
            controls: (batch, seq_len, max_control_dim) - predicted controls (masked if control_mask provided)
        """
        batch_size, seq_len, _ = input_sequence.shape
        device = input_sequence.device
        
        # Embed input (states + dimension encoding)
        x = self.state_embedding(input_sequence)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        # Create causal mask for autoregressive prediction
        causal_mask = self.create_causal_mask(seq_len, device)  # (1, seq_len, seq_len)
        
        # Combine with padding mask if provided
        if padding_mask is not None:
            # Expand padding mask to attention shape
            # padding_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            # Combine with causal mask
            attention_mask = causal_mask.unsqueeze(0) * padding_mask
        else:
            attention_mask = causal_mask
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Predict controls
        controls = self.control_head(x)  # (batch, seq_len, max_control_dim)
        
        # Apply control mask if provided (zero out inactive control dimensions)
        if control_mask is not None:
            controls = controls * control_mask.unsqueeze(1)  # (batch, seq_len, max_control_dim)
        
        return controls
    
    def predict_control(self, state_sequence, n_u, n_x):
        """
        Predict control for a single state sequence (inference).
        
        Args:
            state_sequence: (seq_len, state_dim) - state history
            n_u: int - actual control dimension for this system
            n_x: int - actual state dimension for this system
        
        Returns:
            control: (n_u,) - predicted control at current time (only active dimensions)
        """
        self.eval()
        with torch.no_grad():
            seq_len, state_dim = state_sequence.shape
            device = state_sequence.device
            
            # Create dimension encoding (binary)
            n_u_bits = self._int_to_binary(n_u, 3)  # 3 bits for n_u
            n_x_bits = self._int_to_binary(n_x, 4)  # 4 bits for n_x
            dim_encoding = torch.tensor(n_u_bits + n_x_bits, dtype=torch.float32, device=device)
            
            # Pad state sequence to max dimension
            if state_dim < self.max_state_dim:
                padding = torch.zeros(seq_len, self.max_state_dim - state_dim, device=device)
                state_sequence_padded = torch.cat([state_sequence, padding], dim=1)
            else:
                state_sequence_padded = state_sequence
            
            # Concatenate dimension encoding to each timestep
            dim_encoding_repeated = dim_encoding.unsqueeze(0).repeat(seq_len, 1)  # (seq_len, 7)
            input_sequence = torch.cat([state_sequence_padded, dim_encoding_repeated], dim=1)  # (seq_len, input_dim)
            
            # Add batch dimension
            input_sequence = input_sequence.unsqueeze(0)  # (1, seq_len, input_dim)
            
            # Create control mask (only first n_u dimensions are active)
            control_mask = torch.zeros(1, self.max_control_dim, device=device)
            control_mask[0, :n_u] = 1.0
            
            # Forward pass
            controls = self.forward(input_sequence, control_mask=control_mask)  # (1, seq_len, max_control_dim)
            
            # Return control at last time step (only active dimensions)
            control = controls[0, -1, :n_u]  # (n_u,)
            
        return control
    
    def _int_to_binary(self, num, n_bits):
        """Convert integer to binary list."""
        binary = [int(x) for x in format(num, f'0{n_bits}b')]
        return binary


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    print("=" * 70)
    print("Testing Universal LQR Transformer with Dimension Encoding")
    print("=" * 70)
    
    # Create model
    max_state_dim = 12
    max_control_dim = 6
    dimension_encoding_size = 7  # 3 bits for n_u + 4 bits for n_x
    
    model = UniversalLQRTransformer(
        max_state_dim=max_state_dim,
        max_control_dim=max_control_dim,
        dimension_encoding_size=dimension_encoding_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=128
    )
    
    print(f"\n📊 Model Configuration:")
    print(f"   Max state dim: {max_state_dim}")
    print(f"   Max control dim: {max_control_dim}")
    print(f"   Dimension encoding size: {dimension_encoding_size}")
    print(f"   Input dim: {model.input_dim}")
    print(f"   Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    print(f"\n🧪 Test 1: Batch forward pass")
    batch_size = 32
    seq_len = 64
    input_dim = max_state_dim + dimension_encoding_size
    
    # Simulate input: padded states + dimension encoding
    input_sequence = torch.randn(batch_size, seq_len, input_dim)
    
    # Create control mask (simulate different systems with different n_u)
    control_mask = torch.ones(batch_size, max_control_dim)
    # For example, first half has n_u=4, second half has n_u=6
    control_mask[:batch_size//2, 4:] = 0  # Mask out last 2 dims for first half
    
    controls = model(input_sequence, control_mask=control_mask)
    
    print(f"   Input shape: {input_sequence.shape}")
    print(f"   Output shape: {controls.shape}")
    print(f"   Control mask shape: {control_mask.shape}")
    print("   ✓ Batch forward pass successful!")
    
    # Test inference
    print(f"\n🧪 Test 2: Single sequence inference")
    test_seq_len = 50
    test_state_dim = 4  # e.g., CartPole
    test_n_u = 1
    test_n_x = 4
    
    state_sequence = torch.randn(test_seq_len, test_state_dim)
    control = model.predict_control(state_sequence, n_u=test_n_u, n_x=test_n_x)
    
    print(f"   State sequence shape: {state_sequence.shape}")
    print(f"   Predicted control shape: {control.shape}")
    print(f"   Expected control dim: {test_n_u}")
    print("   ✓ Inference successful!")
    
    # Test with different system
    print(f"\n🧪 Test 3: Different system (6-DOF manipulator)")
    test_state_dim = 12
    test_n_u = 6
    test_n_x = 12
    
    state_sequence = torch.randn(test_seq_len, test_state_dim)
    control = model.predict_control(state_sequence, n_u=test_n_u, n_x=test_n_x)
    
    print(f"   State sequence shape: {state_sequence.shape}")
    print(f"   Predicted control shape: {control.shape}")
    print(f"   Expected control dim: {test_n_u}")
    print("   ✓ Inference successful!")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)

