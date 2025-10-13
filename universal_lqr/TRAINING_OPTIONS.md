# Training Options for 85M Samples

You have **85,794,024 sequences** to train on. Here are your options:

## Option 1: Fast CPU Training (RECOMMENDED) ⚡

Use the optimized CPU config with a smaller model:

```bash
# Edit train_jax.py line 19 to use fast config:
from config_fast_cpu import *  # Instead of: from config import *

# Then train:
python train_jax.py
```

**Specs:**
- Model: ~40k parameters (vs 200k)
- Layers: 2 (vs 4)
- Batch size: 512
- Epochs: 20
- **Expected time: 12-20 hours** ⚡

**Trade-off:** Smaller model might have slightly lower performance, but much faster training.

---

## Option 2: Full Model CPU Training (SLOW) 🐌

Keep current config but accept it will be VERY slow:

```bash
python train_jax.py  # With current config.py
```

**Specs:**
- Model: 208k parameters
- Layers: 4
- Batch size: 256
- Epochs: 50
- **Expected time: ~15 days** 🐌

**Trade-off:** Better model performance, but impractical training time.

---

## Option 3: Cloud GPU (FASTEST & RECOMMENDED) 🚀

Rent a GPU instance for a few dollars:

### Google Colab Pro ($10/month)
```bash
# Upload your data to Google Drive
# Run training on T4/A100 GPU
# Expected time: 2-6 hours
# Cost: ~$0.50 per training run
```

### AWS/Azure GPU Instance
```bash
# g4dn.xlarge (T4 GPU): $0.526/hour
# Expected time: 4-8 hours
# Total cost: $2-4
```

**This is what your friend likely used!**

---

## Option 4: Subset Training (TEST) 🧪

Create a small subset to test your pipeline:

```python
# Create 1M sample test dataset:
python -c "
import h5py
import numpy as np

# Sample 1M from 85M
indices = np.sort(np.random.choice(85_794_024, 1_000_000, replace=False))

with h5py.File('data/training_data_jax.h5', 'r') as f_in:
    with h5py.File('data/training_data_1M.h5', 'w') as f_out:
        # Copy sampled data
        f_out.create_dataset('input_sequences', data=f_in['input_sequences'][indices])
        f_out.create_dataset('controls', data=f_in['controls'][indices])
        f_out.create_dataset('control_masks', data=f_in['control_masks'][indices])
        
        # Copy metadata
        for key in f_in.attrs:
            f_out.attrs[key] = f_in.attrs[key]
        f_out.attrs['total_sequences'] = 1_000_000
        
        # Copy normalization stats
        for key in ['state_mean', 'state_std', 'control_mean', 'control_std']:
            f_out.create_dataset(key, data=f_in[key][:])

print('Created training_data_1M.h5')
"

# Update train_jax.py to use training_data_1M.h5
# Expected time: ~20 minutes
```

---

## Recommendation 🎯

**For quick results:**
1. Use **Option 1 (Fast CPU)** if you want results today
2. Or use **Option 3 (Cloud GPU)** if you can spend $2-4

**For best model:**
1. Test with **Option 4 (1M subset)** first to verify pipeline
2. Then use **Option 3 (Cloud GPU)** for full training

**Why your friend was faster:**
- Likely used GPU (10-20x faster)
- Possibly smaller model (40k params vs your 200k)
- Maybe PyTorch (more CPU-optimized than JAX)

---

## Quick Start (Option 1)

```bash
cd jax_implementation

# 1. Edit train_jax.py line 19:
#    Change: from config import *
#    To:     from config_fast_cpu import *

# 2. Train:
python train_jax.py

# Expected: ~15 hours on M4 CPU
```

Would you like me to help you set up any of these options?

