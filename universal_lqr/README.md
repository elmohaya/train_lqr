# Universal LQR Transformer

A transformer-based neural network that learns to stabilize and control **any** unseen Linear Time-Invariant (LTI) system through LQR demonstrations.

## Goal

Train a single transformer model that can:
- Stabilize any LTI system without prior knowledge of system dynamics
- Generalize across different system types (mechanical, electrical, aerospace, etc.)
- Handle parameter variations and uncertainties
- Predict optimal control inputs given only state trajectories

## System Architecture

### 1. Data Generation
- **33+ LTI system families** covering:
  - Mechanical systems (cart-pole, pendulums, flexible beams, etc.)
  - Electrical systems (DC/AC motors, power converters, RLC circuits)
  - Robotics (manipulators, mobile robots)
  - Vehicles (lateral dynamics, cruise control, platooning)
  - Aerospace (quadrotors, fixed-wing, VTOL)
  - Other systems (chemical reactors, biological models)

- **Parameter variations**: 10 variants per system family with ±30% uncertainty
- **Process noise**: Realistic disturbances and noise in states
- **LQR control**: Optimal LQR controllers designed for each system
- **Uncertain dynamics**: LQR designed on nominal system, data from uncertain system

### 2. Transformer Architecture
- **GPT-style decoder**: Autoregressive prediction of controls
- **Multi-head attention**: 8 attention heads
- **6 transformer layers** with 256 embedding dimensions
- **System-agnostic**: No system labels, learns purely from state-control patterns
- **Task**: Given state history x(0:t), predict control u(t)

### 3. Training
- **Standardization**: Zero mean, unit variance normalization
- **Sequence length**: 64 timesteps (adjustable)
- **Batch size**: 128
- **Optimizer**: AdamW with cosine annealing
- **Loss**: MSE between predicted and LQR controls

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

All adjustable parameters are in `config.py`:

```python
# Data generation
SAVE_NEW_DATA = True              # Set to False to use existing data
NUM_TRAJECTORIES_PER_VARIANT = 100
TIME_HORIZON = 20.0               # seconds
DT = 0.02                         # 50 Hz sampling
NUM_VARIANTS_PER_SYSTEM = 10
PARAMETER_UNCERTAINTY = 0.30      # ±30%
PROCESS_NOISE_STD = 0.01

# Transformer
SEQUENCE_LENGTH = 64              # Context window

# Training
TRAINING_CONFIG = {
    'batch_size': 128,
    'learning_rate': 1e-4,
    'num_epochs': 100,
}
```

### Usage

#### Option 1: Run everything
```bash
python main.py --mode all
```

#### Option 2: Step by step

**Generate data:**
```bash
python data_generation.py
```

**Train transformer:**
```bash
python train.py
```

**Evaluate on test systems:**
```bash
python evaluate.py
```

#### Option 3: Use existing data
```python
# In config.py, set:
SAVE_NEW_DATA = False

# Then run:
python main.py --mode train
```

## Project Structure

```
universal_lqr/
├── config.py                 # All configuration parameters
├── main.py                   # Main execution script
├── lqr_controller.py         # LQR design and stability verification
├── data_generation.py        # LQR data generation pipeline
├── transformer_model.py      # GPT-style transformer architecture
├── train.py                  # Training script
├── evaluate.py               # Evaluation and comparison
├── requirements.txt          # Python dependencies
├── systems/                  # LTI system implementations
│   ├── __init__.py
│   ├── base_system.py       # Base class for all systems
│   ├── mechanical_systems.py
│   ├── electrical_systems.py
│   ├── robotics_systems.py
│   ├── vehicle_systems.py
│   ├── aerospace_systems.py
│   └── other_systems.py
├── data/                     # Generated training data
├── models/                   # Saved model checkpoints
└── logs/                     # Training logs and plots
```

## LTI Systems Implemented

### Mechanical Systems (13)
- Mass-Spring-Damper
- Simple Pendulum
- Inverted Pendulum
- Double Pendulum
- Cart-Pole
- Acrobot
- Furuta Pendulum
- Ball and Beam
- Ball and Plate
- Reaction Wheel Pendulum
- Flexible Beam
- Magnetic Levitation
- Suspension System

### Electrical Systems (8)
- DC Motor
- AC Motor
- Buck Converter
- Boost Converter
- Buck-Boost Converter
- Inverter
- RLC Circuit (Series)
- RLC Circuit (Parallel)

### Robotics Systems (4)
- Two-Link Arm
- Three-Link Manipulator
- Unicycle Robot
- Differential Drive Robot

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

**Total: 34 system families × 10 variants × 100 trajectories = 34,000 trajectories**

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

## Expected Behavior

1. **Data Generation**: ~30-60 minutes for all systems
2. **Training**: Depends on GPU (1-4 hours on modern GPU)
3. **Performance**: Transformer should achieve comparable or better performance than LQR on trained systems and generalize to new systems

## Evaluation

The `evaluate.py` script compares:
- LQR controller (optimal for known system)
- Transformer controller (zero-shot on potentially unseen system)

Metrics:
- State cost: Σ||x(t)||²
- Control cost: Σ||u(t)||²
- Visual comparison of trajectories

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

