# Architecture Quick Reference

## 📐 Dimensions

```python
MAX_STATE_DIM = 12          # Maximum n_x across all systems
MAX_INPUT_DIM = 6           # Maximum n_u across all systems
N_U_ENCODING_BITS = 3       # Binary encoding for n_u (0-6)
N_X_ENCODING_BITS = 4       # Binary encoding for n_x (0-12)
DIMENSION_ENCODING_SIZE = 7 # Total: 3 + 4 bits
```

---

## 🔄 Data Format

### Input (per timestep):
```
[x_padded, n_u_binary, n_x_binary]
└── 12 ──┘ └─ 3 ─┘ └── 4 ──┘
        = 19 dimensions total
```

### Output:
```
u_padded (6 dimensions)
└─ n_u ─┘└─ padding ─┘
   active  (masked)
```

---

## 🛠️ Key Functions

### Data Preprocessing (`data_utils.py`)

```python
# Create dimension encoding
encoding = create_dimension_encoding(n_u, n_x)
# Returns: array of shape (7,) with binary values

# Pad states
states_padded = pad_state(states, target_dim=12)
# Input: (seq_len, n_x) → Output: (seq_len, 12)

# Pad controls
controls_padded = pad_control(controls, target_dim=6)
# Input: (seq_len, n_u) → Output: (seq_len, 6)

# Prepare full input sequence
input_seq = prepare_input_sequence(states, n_u, n_x)
# Input: (seq_len, n_x) → Output: (seq_len, 19)

# Prepare training batch
batch = prepare_training_batch(
    state_sequences, control_sequences, n_u_list, n_x_list
)
# Returns dict with:
#   - input_sequences: (batch, seq_len, 19)
#   - target_controls: (batch, seq_len, 6)
#   - control_masks: (batch, 6)
#   - padding_masks: (batch, seq_len)
```

### Model (`transformer_model.py`)

```python
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

# Forward pass (training)
predictions = model(
    input_sequences,           # (batch, seq_len, 19)
    padding_mask=padding_mask, # (batch, seq_len)
    control_mask=control_mask  # (batch, 6)
)
# Returns: (batch, seq_len, 6)

# Inference
control = model.predict_control(
    state_sequence,  # (seq_len, n_x)
    n_u=n_u,        # int
    n_x=n_x         # int
)
# Returns: (n_u,) - only active dimensions
```

---

## 🎯 Masking

### Control Mask
```python
# For a system with n_u=3
control_mask = [1, 1, 1, 0, 0, 0]
               └─ active ─┘└─ inactive ─┘

# Applied during loss computation:
loss = MSE(pred * mask, target * mask)
```

### Padding Mask
```python
# For sequence of length 50 in batch with max_len=64
padding_mask = [1, 1, ..., 1, 0, ..., 0]
               └── 50 ones ──┘└─ 14 zeros ─┘
```

---

## 📊 Example Systems

### CartPole (Simple)
```python
n_x = 4, n_u = 1
State: [x, theta, x_dot, theta_dot]
Control: [force]

# Processing:
State padded: [x, theta, x_dot, theta_dot, 0, 0, 0, 0, 0, 0, 0, 0]
Encoding: [0, 0, 1, 0, 1, 0, 0]  # n_u=1, n_x=4 in binary
Input: 19 dimensions
Output: [u, 0, 0, 0, 0, 0] → use only first dimension
```

### 6-DOF Manipulator (Complex)
```python
n_x = 12, n_u = 6
State: [q1, ..., q6, q1_dot, ..., q6_dot]
Control: [tau1, tau2, tau3, tau4, tau5, tau6]

# Processing:
State padded: [q1, ..., q6_dot]  # Already 12, no padding
Encoding: [1, 1, 0, 1, 1, 0, 0]  # n_u=6, n_x=12 in binary
Input: 19 dimensions
Output: [u1, u2, u3, u4, u5, u6] → use all dimensions
```

---

## 🔢 Binary Encoding Examples

```python
n_u = 1  → [0, 0, 1]  # 3 bits
n_u = 3  → [0, 1, 1]
n_u = 6  → [1, 1, 0]

n_x = 4  → [0, 1, 0, 0]  # 4 bits
n_x = 6  → [0, 1, 1, 0]
n_x = 12 → [1, 1, 0, 0]
```

---

## 🧪 Quick Test

```python
# Test the utilities
python data_utils.py

# Test the model
python transformer_model.py

# Both should show ✓ All tests passed!
```

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `config.py` | All configuration parameters |
| `transformer_model.py` | Model definition |
| `data_utils.py` | Preprocessing utilities |
| `data_generation.py` | Generate LQR training data |
| `train.py` | Training loop |
| `evaluate.py` | Evaluation scripts |

---

## 🚀 Workflow

1. **Generate Data**
   ```bash
   python main.py --mode generate
   ```

2. **Train Model**
   ```bash
   python main.py --mode train
   ```

3. **Evaluate**
   ```bash
   python main.py --mode evaluate
   ```

4. **All-in-One**
   ```bash
   python main.py --mode all
   ```

---

## 💡 Key Design Principles

1. ✅ **Single unified model** for all systems
2. ✅ **System agnostic** - no explicit labels
3. ✅ **Binary encoding** tells model what dims are active
4. ✅ **Padding + masking** for efficiency
5. ✅ **Batch mixing** for better generalization

---

## 📊 Expected Dataset

```
Systems:              35
Variants per system:  10
Trajectories/variant: 100
──────────────────────────
Total trajectories:   35,000

Distribution by dimension:
  n_u=1: 21 systems (60%)
  n_u=2:  6 systems (17%)
  n_u=3:  4 systems (11%)
  n_u=4:  3 systems (9%)
  n_u=6:  1 system  (3%)
```

---

## 🎯 Model Parameters

```
Architecture:
  d_model: 256
  n_heads: 8
  n_layers: 6
  d_ff: 1024
  dropout: 0.1

Total Parameters: ~5 million
```

---

For detailed documentation, see:
- `TRANSFORMER_ARCHITECTURE.md` - Complete architecture guide
- `SYSTEM_BANK_FINAL.md` - System bank overview
- `QUICK_REFERENCE.md` - General project reference

