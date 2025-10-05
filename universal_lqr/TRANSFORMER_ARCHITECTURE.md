# Universal LQR Transformer Architecture

## Overview

This document describes the transformer architecture designed to learn universal LQR control policies across variable-dimensional LTI systems.

---

## 🎯 Key Design Goals

1. **System Agnostic**: No explicit system labels or metadata
2. **Variable Dimensions**: Handle systems with different state/control dimensions
3. **Efficient**: Use padding and masking rather than separate models
4. **Generalizable**: Learn from diverse systems and transfer to unseen ones

---

## 📊 System Dimensions

Across our 35 LTI systems:

| Dimension | Range | Distribution |
|-----------|-------|--------------|
| **State (n_x)** | 2-12 | Most common: 4 states (14 systems) |
| **Control (n_u)** | 1-6 | Most common: 1 input (21 systems) |

### Maximum Dimensions:
- **MAX_STATE_DIM = 12** (QuadrotorHover, SixDOFManipulator)
- **MAX_INPUT_DIM = 6** (SixDOFManipulator only)

---

## 🏗️ Architecture Components

### 1. **Input Representation**

Each timestep input consists of:
```
Input(t) = [x_padded(t), n_u_binary, n_x_binary]
```

Where:
- `x_padded(t)`: State vector padded to `MAX_STATE_DIM = 12`
- `n_u_binary`: 3-bit binary encoding of control dimension (0-6)
- `n_x_binary`: 4-bit binary encoding of state dimension (0-12)

**Total Input Dimension**: `12 + 3 + 4 = 19`

#### Example:

For **CartPole** (n_x=4, n_u=1):
```python
State: [x, theta, x_dot, theta_dot]  # 4 dimensions
Padded: [x, theta, x_dot, theta_dot, 0, 0, 0, 0, 0, 0, 0, 0]  # 12 dimensions
n_u encoding: [0, 0, 1]  # Binary for 1
n_x encoding: [0, 1, 0, 0]  # Binary for 4
Input: [...padded state..., 0, 0, 1, 0, 1, 0, 0]  # 19 dimensions
```

For **SixDOFManipulator** (n_x=12, n_u=6):
```python
State: [q1, ..., q6, q1_dot, ..., q6_dot]  # 12 dimensions
Padded: [q1, ..., q6, q1_dot, ..., q6_dot]  # Already 12, no padding needed
n_u encoding: [1, 1, 0]  # Binary for 6
n_x encoding: [1, 1, 0, 0]  # Binary for 12
Input: [...state..., 1, 1, 0, 1, 1, 0, 0]  # 19 dimensions
```

---

### 2. **Dimension Encoding Rationale**

**Why binary encoding?**
1. **Efficient**: Only 7 extra dimensions (vs. one-hot would need 6+12=18)
2. **Generalizable**: Binary representation may help with interpolation between dimensions
3. **Informative**: Transformer can learn from bit patterns

**Why concatenate to every timestep?**
- Constant reminder to the transformer which dimensions are active
- Allows attention mechanism to condition on system dimension
- Simple and effective (alternative: global token would lose temporal alignment)

---

### 3. **Output Representation**

Output is a **fixed-size control vector**:
```
Output(t) = u_padded(t)  # Shape: (MAX_INPUT_DIM,) = (6,)
```

During training:
- Target controls are padded to `MAX_INPUT_DIM = 6`
- Control mask zeros out inactive dimensions in loss computation

During inference:
- Model outputs 6 dimensions
- Only first `n_u` dimensions are used
- Rest are ignored

#### Example:

For **CartPole** (n_u=1):
```python
Model output: [u1, 0.02, -0.15, 0.31, -0.08, 0.19]  # 6 values
Control mask: [1, 0, 0, 0, 0, 0]  # Only first is active
Actual control: u1  # Use only first dimension
```

For **Quadrotor** (n_u=4):
```python
Model output: [u1, u2, u3, u4, 0.05, -0.12]  # 6 values
Control mask: [1, 1, 1, 1, 0, 0]  # First 4 are active
Actual control: [u1, u2, u3, u4]  # Use first 4 dimensions
```

---

### 4. **Transformer Model**

```
UniversalLQRTransformer:
  Input: (batch, seq_len, 19)
  ↓
  Linear Embedding: 19 → 256 (d_model)
  ↓
  Positional Encoding
  ↓
  Transformer Blocks (6 layers):
    - Multi-Head Self-Attention (8 heads)
    - Causal Mask (autoregressive)
    - Feed-Forward Network (256 → 1024 → 256)
    - Layer Normalization
    - Residual Connections
  ↓
  Output Head: 256 → 1024 → 6 (MAX_INPUT_DIM)
  ↓
  Output: (batch, seq_len, 6)
```

**Total Parameters**: ~5 million

---

### 5. **Masking Strategy**

#### A. **Padding Mask** (for sequences)

Masks out padded timesteps when sequences have different lengths.

```python
# If sequence lengths are [50, 64, 45] in a batch with max_len=64
padding_mask = [
    [1,1,...,1, 0,0,...,0],  # 50 ones, 14 zeros
    [1,1,...,1],             # 64 ones, 0 zeros
    [1,1,...,1, 0,0,...,0],  # 45 ones, 19 zeros
]  # Shape: (batch, seq_len)
```

#### B. **Control Mask** (for outputs)

Masks out inactive control dimensions.

```python
# For systems with n_u = [1, 3, 6] in a batch
control_mask = [
    [1, 0, 0, 0, 0, 0],  # CartPole (n_u=1)
    [1, 1, 1, 0, 0, 0],  # 3-input system
    [1, 1, 1, 1, 1, 1],  # SixDOFManipulator (n_u=6)
]  # Shape: (batch, MAX_INPUT_DIM)
```

Applied during loss computation:
```python
loss = ((predictions - targets) ** 2) * control_mask
loss = loss.sum() / control_mask.sum()  # Average only over active dims
```

#### C. **Causal Mask** (for attention)

Prevents the model from attending to future timesteps (autoregressive).

```python
# For sequence length = 4
causal_mask = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]  # Shape: (seq_len, seq_len)
```

---

## 🔄 Data Flow

### Training

```
1. Load trajectory: (t, X, U) where X: (T, n_x), U: (T, n_u)

2. Pad states:
   X_padded = pad(X, MAX_STATE_DIM)  # (T, 12)

3. Create dimension encoding:
   dim_enc = [n_u_binary, n_x_binary]  # (7,)

4. Concatenate:
   Input = [X_padded, dim_enc]  # (T, 19)

5. Pad controls:
   U_padded = pad(U, MAX_INPUT_DIM)  # (T, 6)

6. Create masks:
   control_mask = [1,...,1, 0,...,0]  # (6,) with n_u ones

7. Forward pass:
   U_pred = Transformer(Input, control_mask)  # (T, 6)

8. Compute loss (only on active dims):
   loss = MSE(U_pred * control_mask, U_padded * control_mask)
```

### Inference

```
1. Given: state history X: (t, n_x) and dimensions (n_u, n_x)

2. Prepare input:
   Input = prepare_input_sequence(X, n_u, n_x)  # (t, 19)

3. Forward pass:
   U_pred = model.predict_control(X, n_u, n_x)  # (n_u,)

4. Use control:
   u(t) = U_pred  # Only first n_u dimensions
```

---

## 📈 Training Strategy

### Loss Function

```python
def masked_mse_loss(predictions, targets, control_mask):
    """
    Compute MSE only over active control dimensions.
    
    Args:
        predictions: (batch, seq_len, MAX_INPUT_DIM)
        targets: (batch, seq_len, MAX_INPUT_DIM)
        control_mask: (batch, MAX_INPUT_DIM)
    
    Returns:
        loss: scalar
    """
    # Expand mask to match prediction shape
    mask = control_mask.unsqueeze(1)  # (batch, 1, MAX_INPUT_DIM)
    
    # Compute squared error
    squared_error = (predictions - targets) ** 2
    
    # Apply mask
    masked_error = squared_error * mask
    
    # Average over active dimensions only
    loss = masked_error.sum() / mask.sum()
    
    return loss
```

### Batch Composition

**Strategy**: Mix systems in each batch for better generalization

```python
# Example batch of size 32:
# - 10 samples from 2-state systems
# - 12 samples from 4-state systems  
# - 6 samples from 6-state systems
# - 4 samples from 12-state systems

# This ensures the model sees diverse dimensions in each batch
```

### Data Augmentation

1. **Random initial conditions**: Each trajectory starts from different x₀
2. **Parameter variations**: 10 variants per system (±30% parameters)
3. **Process noise**: Small random perturbations during simulation
4. **Sequence windowing**: Extract multiple subsequences from long trajectories

---

## 🎓 Why This Design?

### Alternative Approaches Considered

#### 1. **Separate Models per Dimension**
❌ **Rejected**: Would need 35+ separate models, no transfer learning

#### 2. **One-Hot Encoding of Dimensions**
❌ **Rejected**: Too many extra dimensions (18 vs 7), less generalizable

#### 3. **System ID Tokens**
❌ **Rejected**: Not system-agnostic, wouldn't generalize to new systems

#### 4. **Graph Neural Networks**
❌ **Rejected**: LTI systems don't have clear graph structure

#### 5. **Variable-Size Architecture (Dynamic)**
❌ **Rejected**: Complex to implement, hard to batch efficiently

### Our Approach: **Padding + Masking + Dimension Encoding** ✅

**Advantages:**
1. ✅ **Single unified model** for all systems
2. ✅ **System agnostic** - no explicit labels
3. ✅ **Efficient batching** - fixed dimensions
4. ✅ **Transfer learning** - shared representations
5. ✅ **Simple implementation** - standard transformer
6. ✅ **Generalizable** - binary encoding may interpolate

---

## 📊 Comparison to Standard Transformers

| Aspect | Standard Transformer | Our Universal LQR Transformer |
|--------|---------------------|------------------------------|
| **Input** | Token embeddings | Padded states + dim encoding |
| **Output** | Token probabilities | Control values |
| **Masking** | Padding only | Padding + control mask |
| **Task** | Sequence modeling | Control prediction |
| **Variable dim?** | ❌ Fixed vocabulary | ✅ Handles 1-6 inputs, 2-12 states |

---

## 🧪 Testing & Validation

### Unit Tests (Completed ✅)

1. ✅ **Dimension encoding**: Binary representation correct
2. ✅ **Padding**: States and controls padded to correct size
3. ✅ **Masking**: Control masks created correctly
4. ✅ **Forward pass**: Model runs with variable dimensions
5. ✅ **Inference**: predict_control works for different systems

### Integration Tests (To Do)

- [ ] Train on subset, test on held-out systems
- [ ] Verify control mask prevents learning on inactive dims
- [ ] Test generalization to new system dimensions
- [ ] Validate LQR performance metrics

---

## 📁 File Structure

```
├── config.py                 # All configurable parameters
│   └── MAX_STATE_DIM = 12
│   └── MAX_INPUT_DIM = 6
│   └── DIMENSION_ENCODING_SIZE = 7
│
├── transformer_model.py      # Model definition
│   └── UniversalLQRTransformer
│   └── MultiHeadAttention
│   └── PositionalEncoding
│
├── data_utils.py            # Preprocessing utilities
│   └── prepare_input_sequence()
│   └── create_dimension_encoding()
│   └── create_control_mask()
│   └── prepare_training_batch()
│
└── data_generation.py       # Generate LQR training data
    └── generate_system_variants()
    └── generate_trajectory()
```

---

## 🚀 Usage Example

### Create and Test Model

```python
from transformer_model import UniversalLQRTransformer
from data_utils import prepare_input_sequence
import torch

# Create model
model = UniversalLQRTransformer(
    max_state_dim=12,
    max_control_dim=6,
    dimension_encoding_size=7,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    dropout=0.1
)

# Test with CartPole (n_x=4, n_u=1)
states = torch.randn(50, 4)  # 50 timesteps
n_u, n_x = 1, 4

control = model.predict_control(states, n_u, n_x)
print(f"Predicted control: {control}")  # Shape: (1,)
```

### Training Loop (Simplified)

```python
from data_utils import prepare_training_batch

# Prepare batch (mixed systems)
batch = prepare_training_batch(
    state_sequences=[...],  # List of state trajectories
    control_sequences=[...],  # List of control trajectories
    n_u_list=[1, 3, 4, 6],  # Different control dimensions
    n_x_list=[4, 6, 12, 12]  # Different state dimensions
)

# Forward pass
predictions = model(
    batch['input_sequences'],
    control_mask=batch['control_masks']
)

# Compute loss (only on active dimensions)
loss = masked_mse_loss(
    predictions, 
    batch['target_controls'],
    batch['control_masks']
)

# Backward pass
loss.backward()
optimizer.step()
```

---

## 🎯 Expected Performance

### Generalization Scenarios

1. **Seen system, new parameters**: Should perform near LQR optimal
2. **Seen system, new initial conditions**: Should perform near LQR optimal
3. **Unseen system, similar dimensions**: Good performance expected
4. **Unseen system, novel dimensions**: Moderate performance expected

### Metrics

- **Settling time**: How fast does the system reach equilibrium?
- **Control effort**: ||u||² compared to LQR
- **Stability**: Does the system remain stable?
- **Optimality gap**: How close to LQR cost?

---

## 🔮 Future Extensions

1. **Attention analysis**: Visualize what the model attends to
2. **Ablation studies**: Remove dimension encoding, try different encodings
3. **Larger systems**: Test on n_x > 12, n_u > 6
4. **Nonlinear systems**: Extend to nonlinear dynamics (harder!)
5. **Multi-step prediction**: Predict u(t), u(t+1), ...
6. **Uncertainty estimation**: Output control + confidence

---

## 📚 Key Takeaways

1. **Binary dimension encoding** (7 dims) tells the model what dimensions are active
2. **Padding + masking** enables a single model for all dimensions
3. **Control masks** ensure we only learn/predict on active dimensions
4. **System agnostic** - no labels, just dimensions and state sequences
5. **Generalizable** - can potentially handle unseen system dimensions

This architecture balances:
- **Flexibility** (handles variable dimensions)
- **Efficiency** (single model, batched training)
- **Simplicity** (standard transformer with masking)
- **Generalization** (learns universal control patterns)

---

**Ready to train a universal LQR controller! 🚀**

