# LQR Verification Results

This directory contains verification results for all LTI systems in the model bank.

## Contents

### Individual System Plots
Each system has a detailed verification plot showing:
- **State trajectories** over time
- **Control inputs** over time
- **State norm** (log scale) showing convergence
- **Closed-loop eigenvalues** in the complex plane

Files: `<SystemName>_verification.png` (34 files)

### Summary Report
- **`summary_report.png`**: Overall verification statistics
  - Success rate bar chart
  - Eigenvalue distribution
  - Final state norms
  - System dimensions scatter plot

- **`verification_report.txt`**: Detailed text report
  - Overall statistics
  - Individual system results
  - List of failed systems (if any)

## Verification Results Summary

**Last Run:** Generated on latest verification

**Statistics:**
- Total Systems: 34
- LQR Success: 31/34 (91.2%)
- Stable Systems: 31/34 (91.2%)
- Settled Systems: 14/34 (41.2%)

**Failed Systems:**
1. **FurutaPendulum** - LQR design failed
2. **ReactionWheelPendulum** - LQR design failed  
3. **DifferentialDriveRobot** - LQR design failed

> Note: These systems may need better LQR tuning or model adjustments. They will be excluded from training data generation.

## How to Regenerate

```bash
python verify_all_systems.py
```

This takes about 10 seconds to verify all 34 systems.

## Interpreting Results

### ✅ Good System
- LQR design succeeds
- All eigenvalues have negative real parts (stable)
- State norm converges to zero (or small value)
- Control inputs are reasonable

### ⚠️ Problematic System
- LQR design fails → Check system model
- Unstable eigenvalues → Adjust Q/R weights
- State norm doesn't converge → Increase Q weights
- Control saturates → Increase R weights

## Next Steps

After verification:
1. Review failed systems and decide if they need fixing
2. For successful systems, proceed with data generation
3. The data generation pipeline will automatically skip failed systems

