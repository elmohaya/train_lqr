# Getting Started with Universal LQR Transformer

## Overview

This project trains a **single transformer model** that can stabilize **any** Linear Time-Invariant (LTI) system through learning from LQR demonstrations across 34+ system families.

## Project Status

✅ **All components implemented and tested!**

- ✅ 34 LTI system families (mechanical, electrical, robotics, vehicles, aerospace)
- ✅ LQR controller with stability verification
- ✅ Data generation with parameter uncertainty and process noise
- ✅ GPT-style transformer architecture
- ✅ Training pipeline with normalization
- ✅ Evaluation and comparison tools

## Quick Setup (3 Steps)

### Step 1: Install Dependencies

```bash
cd /Users/turki/Desktop/My_PhD/highway_merging/ablation/universal_lqr
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib

### Step 2: Verify Setup

```bash
python test_setup.py
```

This should output "ALL TESTS PASSED!" ✓

### Step 3: Run Training Pipeline

```bash
python main.py --mode all
```

This will:
1. Generate LQR training data for all 34 system families (30-60 min)
2. Train the transformer model (1-4 hours depending on GPU)

## Usage Guide

### Configuration

All parameters are in `config.py`. Key settings:

```python
# Data Generation
SAVE_NEW_DATA = True              # Set False to reuse existing data
NUM_TRAJECTORIES_PER_VARIANT = 100  # Trajectories per variant
TIME_HORIZON = 20.0               # Simulation time (seconds)
DT = 0.02                         # Sampling time (50 Hz)
NUM_VARIANTS_PER_SYSTEM = 10      # Variants per family
PARAMETER_UNCERTAINTY = 0.30      # ±30% uncertainty

# Transformer
SEQUENCE_LENGTH = 64              # Context window size

# Training
TRAINING_CONFIG = {
    'batch_size': 128,
    'learning_rate': 1e-4,
    'num_epochs': 100,
}
```

### Running Components Separately

**1. Generate Data Only:**
```bash
python data_generation.py
```
- Creates `data/lqr_training_data.pkl`
- ~34,000 trajectories (34 families × 10 variants × 100 trajectories)

**2. Train Model Only (using existing data):**
```bash
# First, set in config.py:
SAVE_NEW_DATA = False

# Then train:
python train.py
```

**3. Evaluate Trained Model:**
```bash
python evaluate.py
```
- Compares transformer vs LQR on test systems
- Generates comparison plots in `logs/evaluation/`

### Advanced Usage

**Custom number of trajectories:**
```python
# In config.py
NUM_TRAJECTORIES_PER_VARIANT = 200  # More data
```

**Larger transformer:**
```python
# In config.py
TRANSFORMER_CONFIG = {
    'd_model': 512,      # Larger embedding
    'n_heads': 16,       # More attention heads
    'n_layers': 12,      # Deeper network
    'd_ff': 2048,        # Larger feed-forward
}
```

**Longer training:**
```python
# In config.py
TRAINING_CONFIG = {
    'num_epochs': 200,
}
```

## Understanding the Data Flow

### 1. System Definition
Each system (e.g., CartPole) defines:
- State-space matrices (A, B)
- Default LQR weights (Q, R)
- Parameter sampling methods

### 2. Data Generation Process

For each system:
1. **Create variants** with different parameters (±30%)
2. **Design LQR** on nominal system → optimal gain K
3. **Verify stability** of closed-loop system
4. **Generate trajectories**:
   - LQR uses nominal system
   - Simulation uses uncertain system + noise
   - This creates realistic, suboptimal data

### 3. Transformer Training

Input: State sequence x(0:t) → Output: Control u(t)

The model learns to:
- Recognize system dynamics from state patterns
- Predict stabilizing controls
- Generalize across system types

### 4. Inference

Given a new (potentially unseen) LTI system:
1. Observe state trajectory
2. Transformer predicts control
3. Apply control to system
4. Repeat

## Expected Outputs

### During Data Generation
```
Generating data for 34 system families
  - 10 variants per family
  - 100 trajectories per variant
  ...
✓ LQR controller designed successfully for CartPole
✓ LQR controller designed successfully for InvertedPendulum
...
Data generation complete!
  - Total system variants: 340
  - Total trajectories: 34,000
  - Data saved to: ./data/lqr_training_data.pkl
```

### During Training
```
Dataset created: 340000 samples
  Max state dim: 12
  Max control dim: 4

Model created with 2,345,678 parameters

Epoch 1/100
  Train Loss: 0.234567
  Val Loss: 0.345678
  ✓ Best model saved!
...
```

### During Evaluation
```
Testing: CartPole
  LQR State Cost: 12.3456
  Transformer State Cost: 13.2345
  Ratio: 1.0721
  Plot saved to: logs/evaluation/CartPole_comparison.png
```

## File Structure Explained

```
universal_lqr/
├── config.py              # ← MODIFY THIS for parameters
├── main.py                # ← RUN THIS for full pipeline
│
├── Core Implementation
├── lqr_controller.py      # LQR design & stability verification
├── data_generation.py     # Generate training data
├── transformer_model.py   # GPT-style transformer
├── train.py               # Training loop
├── evaluate.py            # Evaluation & comparison
│
├── System Families
├── systems/
│   ├── mechanical_systems.py   # 13 systems
│   ├── electrical_systems.py   # 8 systems
│   ├── robotics_systems.py     # 4 systems
│   ├── vehicle_systems.py      # 3 systems
│   ├── aerospace_systems.py    # 3 systems
│   └── other_systems.py        # 3 systems
│
├── Generated Artifacts
├── data/                  # Training data (git-ignored)
├── models/                # Saved models (git-ignored)
└── logs/                  # Training logs & plots (git-ignored)
```

## Customization Examples

### Add Your Own System

Create in `systems/mechanical_systems.py` (or appropriate file):

```python
class MyCustomSystem(LTISystem):
    def get_default_params(self):
        return {
            'mass': 1.0,
            'stiffness': 10.0,
            'damping': 0.5
        }
    
    def get_matrices(self):
        m = self.params['mass']
        k = self.params['stiffness']
        c = self.params['damping']
        
        A = np.array([
            [0, 1],
            [-k/m, -c/m]
        ])
        B = np.array([[0], [1/m]])
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([10.0, 1.0])
        R = np.array([[0.1]])
        return Q, R
    
    def sample_initial_condition(self):
        return np.array([
            np.random.uniform(-1, 1),
            np.random.uniform(-0.5, 0.5)
        ])
```

Then add to `systems/__init__.py` and `data_generation.py`.

### Use Pre-Generated Data

```python
# config.py
SAVE_NEW_DATA = False

# Then run
python train.py
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `config.py`:
```python
TRAINING_CONFIG = {
    'batch_size': 64,  # or 32
}
```

### Issue: "LQR controller failed for system X"
**Solution:** This is expected for some parameter variants. The system will skip failed variants and continue.

### Issue: Data generation is slow
**Solution:** 
1. Reduce number of trajectories temporarily for testing:
```python
NUM_TRAJECTORIES_PER_VARIANT = 10
```
2. Or reduce number of variants:
```python
NUM_VARIANTS_PER_SYSTEM = 3
```

### Issue: Training loss not decreasing
**Solution:**
1. Check normalization is enabled (should be by default)
2. Increase learning rate:
```python
TRAINING_CONFIG = {
    'learning_rate': 5e-4,
}
```
3. Train longer or use learning rate warmup

## Performance Tips

### For Faster Data Generation
- Reduce `NUM_VARIANTS_PER_SYSTEM` to 5
- Reduce `NUM_TRAJECTORIES_PER_VARIANT` to 50
- Reduce `TIME_HORIZON` to 10.0

### For Better Model Performance
- Increase `NUM_TRAJECTORIES_PER_VARIANT` to 200
- Increase `SEQUENCE_LENGTH` to 128
- Use larger transformer (more layers/heads)
- Train for more epochs

### For Testing/Development
```python
# Quick test configuration
NUM_VARIANTS_PER_SYSTEM = 2
NUM_TRAJECTORIES_PER_VARIANT = 10
TIME_HORIZON = 5.0

TRANSFORMER_CONFIG = {
    'd_model': 128,
    'n_layers': 3,
}

TRAINING_CONFIG = {
    'num_epochs': 10,
}
```

## Next Steps

After training:

1. **Evaluate performance:**
```bash
python evaluate.py
```

2. **Test on your own system:**
```python
from systems.base_system import LTISystem
from evaluate import load_trained_model, simulate_with_transformer

# Define your system
class MySystem(LTISystem):
    # ... implement required methods

# Load model
model, stats = load_trained_model()

# Test
system = MySystem()
t, X, U = simulate_with_transformer(model, system, stats)
```

3. **Analyze generalization:**
- Test on systems NOT in training set
- Try different parameter ranges
- Add disturbances

## Citation

If you use this code, please cite appropriately in your research.

## Support

For issues or questions:
1. Check this guide
2. Review the README.md
3. Check code comments
4. Open an issue or contact the author

---

**Ready to start? Run:**
```bash
python main.py --mode all
```

Good luck with your research! 🚀

