# LQR System Verification Guide

## Overview

The `verify_all_systems.py` script tests and visualizes LQR control for all 34 system families in the model bank. This ensures that your systems are properly configured before running the full data generation pipeline.

---

## ✨ What It Does

### 1. **Tests Each System**
- Creates system instance with default parameters
- Designs LQR controller using default Q/R weights
- Verifies closed-loop stability (eigenvalue check)
- Simulates 10-second trajectory
- Analyzes convergence and performance

### 2. **Generates Visualizations**
For each system, creates a detailed plot showing:
- **State trajectories** (all states over time)
- **Control inputs** (all controls over time)
- **State norm** (log scale) - shows convergence to origin
- **Eigenvalue plot** - shows stability in complex plane

### 3. **Creates Summary Report**
- **Success rate** statistics
- **Eigenvalue distribution** across all systems
- **Final state norms** for successful systems
- **System dimensions** scatter plot
- **Text report** with detailed results

---

## 🚀 How to Use

### Quick Run
```bash
python verify_all_systems.py
```

### View Results
```bash
# Open summary plot
open verification_results/summary_report.png

# Read text report
cat verification_results/verification_report.txt

# View individual system plot
open verification_results/CartPole_verification.png
```

---

## 📊 Latest Results

### Overall Statistics
```
Total Systems:     34
LQR Success:       31/34 (91.2%)
Stable Systems:    31/34 (91.2%)
Settled Systems:   14/34 (41.2%)
```

### Failed Systems (3)
1. **FurutaPendulum** - LQR design failed (needs better Q/R tuning)
2. **ReactionWheelPendulum** - LQR design failed (needs model adjustment)
3. **DifferentialDriveRobot** - LQR design failed (needs better initialization)

### Successful Systems (31) ✅
All other systems passed verification and are ready for data generation!

---

## 📁 Output Files

All results are saved in `verification_results/`:

### Individual Plots (34 files)
```
CartPole_verification.png
InvertedPendulum_verification.png
QuadrotorHover_verification.png
... (31 more)
```

Each plot contains:
- Top-left: State trajectories
- Top-right: Control inputs
- Bottom-left: State norm (log scale)
- Bottom-right: Eigenvalue plot
- Bottom: System information box

### Summary Files
```
summary_report.png           - Visual summary (4 subplots)
verification_report.txt      - Detailed text report
README.md                    - Results documentation
```

---

## 🔍 Understanding the Plots

### State Trajectories Plot
- **Good:** All states converge to zero (or small value)
- **Bad:** States diverge or oscillate indefinitely

### Control Inputs Plot
- **Good:** Controls decrease over time, reasonable magnitude
- **Bad:** Controls saturate, oscillate, or grow unbounded

### State Norm (Log Scale)
- **Good:** Straight line down to -∞ (exponential convergence)
- **Bad:** Flat line or increasing

### Eigenvalue Plot
- **Stable:** All × marks in left half-plane (green shaded area)
- **Unstable:** Any × marks in right half-plane

---

## 🛠️ Fixing Failed Systems

### If LQR Design Fails

**Option 1: Adjust Q/R Weights**
```python
# In systems/mechanical_systems.py (or appropriate file)
def get_default_lqr_weights(self):
    # Increase state weights (Q) for better regulation
    Q = np.diag([100.0, 10.0])  # Was [10.0, 1.0]
    
    # Decrease input weights (R) for more control effort
    R = np.array([[0.01]])  # Was [[0.1]]
    
    return Q, R
```

**Option 2: Check System Model**
```python
# Verify A and B matrices are correct
def get_matrices(self):
    # ... your model ...
    
    # Debug: print matrices
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"Eigenvalues of A: {np.linalg.eigvals(A)}")
    
    return A, B
```

### If System is Unstable

**Check controllability:**
```python
import numpy as np
from scipy.linalg import ctrb

A, B = system.get_matrices()
C = ctrb(A, B)
rank = np.linalg.matrix_rank(C)

print(f"Controllability matrix rank: {rank}")
print(f"Required rank: {A.shape[0]}")

if rank < A.shape[0]:
    print("System is not controllable!")
```

**Increase Q weights:**
```python
# Higher Q → More aggressive stabilization
Q = np.diag([1000.0, 100.0, ...])  # Increase significantly
```

### If System Doesn't Settle

**Possible causes:**
1. **Too much damping** → Decrease R weights
2. **Not enough state penalty** → Increase Q weights
3. **Model has integrators** → May settle at non-zero value (OK)
4. **Linearization point** → Check equilibrium assumptions

---

## 🎯 Integration with Data Pipeline

The verification script is **independent** but **recommended before** data generation:

### Recommended Workflow
```bash
# 1. Verify all systems (10 seconds)
python verify_all_systems.py

# 2. Review results
open verification_results/summary_report.png

# 3. Fix any failed systems (optional)
# Edit system files, then re-verify

# 4. Generate data (only successful systems)
python data_generation.py
```

### Automatic Handling
The data generation pipeline automatically:
- Tests LQR design for each system
- Skips systems where LQR fails
- Reports failed systems in the output

So verification is **optional but highly recommended** for:
- Understanding which systems work
- Debugging failures before long data generation
- Visual confirmation of proper behavior

---

## 📈 Performance Metrics

### What "Success" Means

| Metric | Criteria | Why It Matters |
|--------|----------|----------------|
| **LQR Success** | Controller computed | System is controllable |
| **Stable** | Max eigenvalue < 0 | Closed-loop won't diverge |
| **Settled** | Final norm < 0.1 | System reaches equilibrium |

### Typical Results

**Fast settling** (< 5 seconds):
- MassSpringDamper
- SimplePendulum
- DCMotor
- DoubleIntegrator

**Moderate settling** (5-10 seconds):
- CartPole
- InvertedPendulum
- QuadrotorHover

**Slow/doesn't settle** (but stable):
- Some platooning models
- Chemical reactor (has offset)
- Some vehicle dynamics

---

## 🔬 Advanced Usage

### Custom Verification Parameters

Edit at the top of `verify_all_systems.py`:
```python
# Verification settings
VERIFICATION_TIME = 20.0  # Increase for slow systems
VERIFICATION_DT = 0.01    # Decrease for faster sampling
```

### Verify Single System

```python
from verify_all_systems import verify_single_system
from systems import CartPole

results = verify_single_system(CartPole, plot=True, verbose=True)

if results['stable']:
    print("System is stable!")
    print(f"Max eigenvalue: {results['max_eigenvalue']}")
```

### Batch Verification of Subset

```python
from verify_all_systems import verify_single_system
from systems import CartPole, InvertedPendulum, QuadrotorHover

systems_to_test = [CartPole, InvertedPendulum, QuadrotorHover]

for sys_class in systems_to_test:
    results = verify_single_system(sys_class)
    print(f"{results['name']}: {'✓' if results['stable'] else '✗'}")
```

---

## 📚 Interpreting the Summary Report

### Plot 1: Success Rate Bar Chart
- Shows counts of LQR success, stable, and settled systems
- Blue dashed line = total systems (34)
- Green bars = all systems passed
- Orange bars = some failed

### Plot 2: Eigenvalue Distribution
- Green histogram = stable systems
- Red histogram = unstable systems (should be empty!)
- Black dashed line = stability boundary (x=0)

### Plot 3: Final State Norm
- Lower is better (closer to origin)
- Log scale shows convergence quality
- Red dashed line = settling threshold (0.1)

### Plot 4: System Dimensions
- Green dots = stable systems
- Red dots = unstable systems
- Shows diversity of system sizes

---

## ❓ FAQ

**Q: Why did some systems fail?**
A: Usually due to poorly tuned Q/R weights or model issues. These can often be fixed.

**Q: Should I fix failed systems before training?**
A: Not necessary - training will use the 31 working systems. But fixing them gives more diverse data.

**Q: What if a system doesn't settle but is stable?**
A: That's OK! Some systems have offsets or very slow dynamics. Stability is what matters.

**Q: How long does verification take?**
A: About 10 seconds total (0.3 seconds per system on average).

**Q: Can I change the LQR weights?**
A: Yes! Edit the `get_default_lqr_weights()` method in the system class.

**Q: What about parameter variants?**
A: This verifies the nominal system. Variants are tested during data generation.

---

## 🚀 Next Steps

After successful verification:

1. **Proceed to data generation:**
   ```bash
   python data_generation.py
   ```

2. **Or run full pipeline:**
   ```bash
   python main.py --mode all
   ```

3. **Or fix failed systems first** (optional):
   - Edit system files
   - Re-run verification
   - Repeat until satisfied

---

## 📞 Support

If verification reveals issues:
1. Check this guide for debugging tips
2. Review system model in code
3. Try different Q/R weights
4. Check system controllability
5. Open an issue if you need help

---

**Happy verifying! 🎯**

The verification ensures your model bank is solid before investing time in data generation and training.

