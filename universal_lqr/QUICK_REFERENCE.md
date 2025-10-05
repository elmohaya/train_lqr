# Universal LQR Transformer - Quick Reference Card

## 🚀 One-Line Execution
```bash
python main.py --mode all
```

## 📋 Common Commands

| Task | Command |
|------|---------|
| **Verify all systems** | `python verify_all_systems.py` ⭐ |
| **Full pipeline** | `python main.py --mode all` |
| **Data only** | `python main.py --mode data` |
| **Training only** | `python main.py --mode train` |
| **Evaluate** | `python evaluate.py` |
| **Test setup** | `python test_setup.py` |
| **Force new data** | `python main.py --mode all --force-data-gen` |

## ⚙️ Quick Config Changes

### Fast Testing (Quick Iteration)
```python
# config.py
NUM_VARIANTS_PER_SYSTEM = 3
NUM_TRAJECTORIES_PER_VARIANT = 20
TIME_HORIZON = 5.0

TRAINING_CONFIG = {'num_epochs': 10, 'batch_size': 64}
```

### Production (Full Training)
```python
# config.py
NUM_VARIANTS_PER_SYSTEM = 10
NUM_TRAJECTORIES_PER_VARIANT = 100
TIME_HORIZON = 20.0

TRAINING_CONFIG = {'num_epochs': 100, 'batch_size': 128}
```

### Use Existing Data
```python
# config.py
SAVE_NEW_DATA = False
```

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| **System families** | 34 |
| **LQR success rate** | **100%** ✅ |
| **Stable systems** | **34/34 (100%)** ✅ |
| **Variants per family** | 10 (configurable) |
| **Trajectories per variant** | 100 (configurable) |
| **Total trajectories** | ~34,000 |
| **Max state dimension** | 12 (Quadrotor) |
| **Max control dimension** | 4 (Quadrotor, BallPlate) |
| **Sequence length** | 64 timesteps |
| **Model parameters** | ~2-3M |
| **Training time (GPU)** | 1-4 hours |
| **Data gen time** | 30-60 min |

## 🎯 System Families by Domain

```
Mechanical (13):    MassSpringDamper, Pendulums (3), CartPole,
                    Acrobot, Furuta, BallBeam, BallPlate,
                    ReactionWheel, FlexibleBeam, MagLev, Suspension

Electrical (8):     DCMotor, ACMotor, Buck/Boost/BuckBoost Converters,
                    Inverter, RLC (2)

Robotics (4):       TwoLinkArm, ThreeLinkManipulator,
                    Unicycle, DifferentialDrive

Vehicles (3):       LateralDynamics, CruiseControl, Platooning

Aerospace (3):      Quadrotor, FixedWing, VTOL

Other (3):          DoubleIntegrator, LotkaVolterra, ChemicalReactor
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `config.py` | **Edit this** for all parameters |
| `main.py` | **Run this** for full pipeline |
| `verify_all_systems.py` | **Verify** all LQR controllers ⭐ |
| `test_setup.py` | **Verify** installation |
| `README.md` | **Read** for details |
| `GETTING_STARTED.md` | **Follow** for setup |

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce `batch_size` to 64 or 32 |
| LQR failures | Normal, system skips failed variants |
| Slow data gen | Reduce `NUM_TRAJECTORIES_PER_VARIANT` |
| No GPU | Training works on CPU (slower) |
| Import errors | Run `pip install -r requirements.txt` |

## 📈 Evaluation Metrics

```python
# evaluate.py computes:
- State cost:   Σ ||x(t)||²
- Control cost: Σ ||u(t)||²  
- Ratio:        Transformer cost / LQR cost

Ideal ratio:    ≈ 1.0 (comparable to optimal LQR)
Good ratio:     < 1.5 (reasonable performance)
```

## 🎓 Add Custom System (3 Steps)

```python
# 1. Define in systems/mechanical_systems.py
class MySystem(LTISystem):
    def get_default_params(self):
        return {'param1': 1.0}
    
    def get_matrices(self):
        A = ... # Define A matrix
        B = ... # Define B matrix
        return A, B
    
    def get_default_lqr_weights(self):
        Q = np.diag([...])
        R = np.diag([...])
        return Q, R
    
    def sample_initial_condition(self):
        return np.random.uniform(...)

# 2. Add to systems/__init__.py
from .mechanical_systems import MySystem
__all__ = [..., 'MySystem']

# 3. Add to data_generation.py
system_classes = [..., MySystem]
```

## 💾 File Locations

```
Data:       ./data/lqr_training_data.pkl
Models:     ./models/best_model.pt
Stats:      ./models/normalization_stats.pkl
Logs:       ./logs/training_history.png
Eval:       ./logs/evaluation/*.png
```

## 🔍 Check Progress

```bash
# Verify all systems first (recommended!)
python verify_all_systems.py
open verification_results/summary_report.png

# During data generation
ls -lh data/

# During training
ls -lh models/
cat logs/training_history.png  # or open in viewer

# After evaluation
ls -lh logs/evaluation/
```

## ⏱️ Time Estimates

| Task | CPU | GPU |
|------|-----|-----|
| Data generation | 30-60 min | 30-60 min |
| Training (100 epochs) | 10-20 hrs | 1-4 hrs |
| Evaluation | 1-2 min | 1-2 min |

## 🎯 Performance Targets

| System Type | Expected Ratio |
|-------------|---------------|
| Trained systems | 0.8 - 1.2 |
| Similar to trained | 1.0 - 1.5 |
| Novel systems | 1.0 - 2.0 |

Ratio = Transformer cost / LQR cost

## 📚 Further Reading

| Topic | File |
|-------|------|
| Installation | GETTING_STARTED.md |
| Architecture | README.md |
| Full details | PROJECT_SUMMARY.md |
| Code docs | Inline comments |

## 🆘 Support

1. Check this card
2. Read GETTING_STARTED.md
3. Review README.md
4. Check code comments
5. Open issue / contact author

---

## 🎯 Recommended Workflow

```bash
# 1. Verify all systems work (takes ~10 seconds)
python verify_all_systems.py

# 2. Check results
open verification_results/summary_report.png
cat verification_results/verification_report.txt

# 3. If satisfied, run full pipeline
python main.py --mode all
```

---

**Ready?** → `python verify_all_systems.py` then `python main.py --mode all` 🚀

