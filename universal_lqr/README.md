# Universal LQR Transformer (JAX/Flax)

A transformer-based neural network that learns to stabilize and control **any** unseen Linear Time-Invariant (LTI) system through LQR demonstrations.

## Goal

Train a single transformer model that can:
- Stabilize any LTI system without prior knowledge of system dynamics
- Generalize across different system types (mechanical, robotics, aerospace, vehicles, etc.)
- Handle parameter variations and uncertainties
- Predict optimal control inputs given only state trajectories

## System Architecture

### 1. Data Generation (NumPy/SciPy)
- **35 LTI system families** covering:
  - Mechanical systems (cart-pole, pendulums, flexible beams, etc.)
  - Robotics (manipulators, mobile robots, quadrupeds, etc.)
  - Vehicles (lateral dynamics, cruise control, platooning)
  - Aerospace (quadrotors, fixed-wing, VTOL)
  - Other systems (chemical reactors, biological models)

- **Parameter variations**: 10 variants per system family with ±10% uncertainty
- **Process noise**: Realistic disturbances and noise in states
- **LQR control**: Optimal LQR controllers designed for each system
- **Uncertain dynamics**: LQR designed on nominal system, data from uncertain system
- **Sequence stride**: Configurable overlap between sequences (stride=16 by default)

### 2. Transformer Architecture (JAX/Flax)
- **GPT-style decoder**: Autoregressive prediction of controls
- **Multi-head attention**: 8 attention heads
- **6 transformer layers** with 256 embedding dimensions
- **System-agnostic**: No system labels, learns purely from state-control patterns
- **Task**: Given state history x(0:t), predict control u(t)
- **Dimension encoding**: Binary encoding of system dimensions concatenated to inputs

### 3. Training (JAX/Flax with Multi-GPU)
- **Framework**: JAX/Flax for maximum GPU performance
- **JIT compilation**: All training/eval steps compiled for speed
- **Standardization**: Zero mean, unit variance normalization
- **Sequence length**: 32 timesteps
- **Batch size**: 1024 (tuned for multi-GPU)
- **Optimizer**: Adam with warmup and cosine decay
- **Loss**: Masked MSE between predicted and LQR controls

## Quick Start

### Installation (Linux Server with NVIDIA GPUs)

```bash
# Install JAX with CUDA support
pip install -r requirements_jax.txt
```

### Configuration

All adjustable parameters are in `config.py`:

```python
# Data generation
SAVE_NEW_DATA = True              # Set to False to use existing data
NUM_TRAJECTORIES_PER_VARIANT = 100
TIME_HORIZON = 50.0               # seconds
DT = 0.02                         # 50 Hz sampling
NUM_VARIANTS_PER_SYSTEM = 10
PARAMETER_UNCERTAINTY = 0.10      # ±10%
PROCESS_NOISE_STD = 0.01
SEQUENCE_STRIDE = 16              # Stride between sequences (16 = 50% overlap)

# Transformer
SEQUENCE_LENGTH = 32              # Context window

# Training
TRAINING_CONFIG = {
    'batch_size': 1024,           # Large batch for multi-GPU
    'learning_rate': 2e-4,
    'num_epochs': 30,
}
```

### Usage (Three-Step Workflow)

**Step 1: Generate preprocessed data (run once):**
```bash
python data_generation.py
```
This creates `data/training_data.h5` (~14 GB) with preprocessed sequences.

**Step 2: Train with JAX (on Linux server with GPUs):**
```bash
python train_jax.py
```
This trains the transformer using JIT-compiled JAX for maximum speed.

**Step 3: Test the trained model:**
```bash
python test_jax.py
```
This evaluates the transformer on test systems and compares with LQR performance.

### Adjusting Speed vs Coverage

Edit `SEQUENCE_STRIDE` in `config.py`:
- `stride=1`: Maximum overlap, 72M sequences, comprehensive training (slowest)
- `stride=8`: High overlap, 9M sequences, thorough training
- `stride=16`: Moderate overlap, 4.5M sequences, balanced
- `stride=32`: No overlap, 2.3M sequences, fastest training

## Project Structure

```
universal_lqr/
├── config.py                 # All configuration parameters
├── lqr_controller.py         # LQR design and stability verification
├── data_generation.py        # Data generation (creates HDF5)
├── train_jax.py              # JAX/Flax training script
├── test_jax.py               # JAX/Flax testing script
├── train_fast.py             # PyTorch training script (alternative)
├── transformer_model.py      # PyTorch transformer architecture
├── data_utils.py             # Data preprocessing utilities
├── verify_all_systems.py     # Verify all LQR controllers
├── requirements_jax.txt      # JAX dependencies for Linux
├── SERVER_SETUP.txt          # Server setup instructions
├── README.md                 # This file
├── systems/                  # LTI system implementations
│   ├── __init__.py
│   ├── base_system.py       # Base class for all systems
│   ├── mechanical_systems.py
│   ├── robotics_systems.py
│   ├── vehicle_systems.py
│   ├── aerospace_systems.py
│   └── other_systems.py
├── data/                     # Generated training data
│   └── training_data.h5     # Preprocessed sequences (~14 GB)
├── models/                   # Saved model checkpoints
│   ├── best_model_jax.pkl   # Best JAX model
│   ├── checkpoint_epoch_*.pkl
│   └── training_history_jax.json
└── test_results/             # Evaluation results
    ├── CartPole_comparison.png
    ├── TwoLinkArm_comparison.png
    └── ...
```

## LTI Systems Implemented

### Mechanical Systems (13)
- Mass-Spring-Damper, Simple Pendulum, Inverted Pendulum
- Double Pendulum, Cart-Pole, Acrobot
- Furuta Pendulum, Ball and Beam, Ball and Plate
- Reaction Wheel Pendulum, Flexible Beam
- Magnetic Levitation, Suspension System

### Electrical Systems (1)
- DC Motor (fast systems removed for diversity)

### Robotics Systems (12)
- Two-Link Arm, Three-Link Manipulator
- Four-Link Manipulator, Five-Link Manipulator
- Two-Joint Planar Robot, Three-Joint Planar Robot
- Unicycle Robot, Differential Drive Robot
- Ackermann Vehicle, Planar Biped
- Cartesian Manipulator, SCARA Robot

### Vehicle Systems (3)
- Vehicle Lateral Dynamics
- Longitudinal Cruise Control
- Platooning Model

### Aerospace Systems (3)
- Quadrotor (Hover)
- Fixed-Wing Aircraft
- VTOL (Linearized)

### Other Systems (3)
- Double Integrator
- Lotka-Volterra (Predator-Prey)
- Chemical Reactor (CSTR)

**Total: 35 system families × 10 variants × 100 trajectories = 35,000 trajectories**
**Sequences: ~4.5M with stride=16 (50% overlap)**

## Key Features

### LQR Stability Verification
Every LQR controller is verified for stability before data generation:
```python
K, success = design_and_verify_lqr(system)
if not success:
    print(f"LQR failed for {system.name}")
```

### Uncertainty Handling
- Nominal system: Used for LQR design
- Uncertain system: Used for trajectory generation (±30% parameter variation)
- Process noise: Applied to states during simulation

### System-Agnostic Learning
The transformer receives:
- **Input**: Padded state sequences (all systems mapped to max dimension)
- **Output**: Padded control predictions
- **No labels**: System type/parameters not provided

### Normalization Strategy
Standardization (zero mean, unit variance) across all dimensions for better training stability.

## Expected Performance

### Data Generation
- **Time**: ~30-60 minutes for all 35 systems × 10 variants × 100 trajectories
- **Output**: `data/training_data.h5` (~14 GB with stride=16)
- **Sequences**: ~4.5M preprocessed transformer-ready sequences

### Training (Linux Server with 3× RTX 2080 Ti)
- **Time**: 5-10 minutes per epoch (30 epochs = 2.5-5 hours total)
- **Speed**: ~15-20k sequences/second with JAX JIT compilation
- **Memory**: Streams batches from HDF5, minimal RAM usage
- **Multi-GPU**: Automatically uses all available GPUs

## Customization

### Add New System Family

1. Create system class in appropriate file (e.g., `systems/mechanical_systems.py`):

```python
class MyNewSystem(LTISystem):
    def get_default_params(self):
        return {'param1': 1.0, 'param2': 2.0}
    
    def get_matrices(self):
        A = ...  # Define state matrix
        B = ...  # Define input matrix
        return A, B
    
    def get_default_lqr_weights(self):
        Q = ...  # State cost
        R = ...  # Input cost
        return Q, R
    
    def sample_initial_condition(self):
        return np.random.uniform(...)
```

2. Add to `systems/__init__.py`
3. Add to `data_generation.py` system list

### Adjust Hyperparameters

Edit `config.py`:
- Increase `NUM_TRAJECTORIES_PER_VARIANT` for more data
- Increase `SEQUENCE_LENGTH` for longer context
- Adjust `TRANSFORMER_CONFIG` for model capacity
- Tune `TRAINING_CONFIG` for better convergence

## Citation

If you use this code in your research, please cite:

```bibtex
@software{universal_lqr_transformer,
  title = {Universal LQR Transformer: Learning to Control Any LTI System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/universal_lqr}
}
```

## License

MIT License

## Contributing

Contributions welcome! Areas of interest:
- Additional system families
- Improved transformer architectures
- Better normalization strategies
- Robustness to out-of-distribution systems
- Real-world validation

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

