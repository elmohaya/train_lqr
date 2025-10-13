# 🔍 Loss Investigation Summary

## Your Questions Answered

### 1. **Are the loss values too big?**
**YES!** The loss of 18,000 was **1,800x too high**.

**Root cause**: Data was not normalized during training.

### 2. **What should be a good loss?**

For normalized MSE loss on control prediction:

| Stage | Expected Loss |
|-------|---------------|
| **Untrained (random)** | 5-10 |
| **Early training (epoch 1-5)** | 2-5 |
| **Mid training (epoch 10)** | 1-3 |
| **Good model (epoch 20)** | 0.5-2.0 |
| **Excellent model (epoch 50)** | < 0.5 |
| **Best possible** | 0.01-0.1 |

### 3. **Should loss be averaged or summed?**

**AVERAGED** (as you're doing correctly):

```python
loss = jnp.sum(masked_error) / jnp.maximum(jnp.sum(masks), 1.0)
```

This gives **MSE per active element**, which is the standard practice. This makes loss:
- Independent of batch size ✅
- Independent of number of active dimensions ✅
- Comparable across different systems ✅

### 4. **Do huge losses affect gradients?**

**YES, massively!**

**Before normalization:**
- Loss: 15,781
- Gradient norms: 60-141
- Gradient clip: 1.0
- **Clipping factor: 60-141x** ❌

This means gradients were reduced by 60-141x, preventing any meaningful learning!

**After normalization:**
- Loss: 8.8
- Gradient norms: 2.8-2.9  
- Gradient clip: 1.0
- **Clipping factor: 2.8-2.9x** ✅

Now gradients are only slightly clipped, allowing proper learning.

### 5. **Are gradients always clipped?**

**Not always, but commonly in transformer training:**

- **Purpose**: Prevent gradient explosion, stabilize training
- **Your setting**: `gradient_clip = 1.0`
- **When it triggers**: When any parameter gradient norm > 1.0
- **Effect**: Scales down ALL gradients to have max norm = 1.0

**With normalized data**: Gradients are 2-3x the clip value → minimal clipping, healthy learning

**With unnormalized data**: Gradients are 60-140x the clip value → severe clipping, no learning

### 6. **Is control input u normalized?**

**It SHOULD be, but it WASN'T!**

**Before fix:**
```python
# Data loader loaded stats but never applied them!
control = f['controls'][indices]  # RAW data: -2686 to +4001
batch = {'controls': control}     # Sent to model unnormalized ❌
```

**After fix:**
```python
# Now normalized in data loader
control = f['controls'][indices]
control = (control - control_mean) / control_std  # Normalized! ✅
batch = {'controls': control}  # Sent to model with mean~0, std~1
```

### 7. **Why did we see values like 3.1 if data was normalized?**

**Two reasons:**

1. **Data wasn't actually normalized** - the bug we just fixed
2. **Normalization stats are for the POPULATION, not guarantees**:
   - Mean ≈ 0, Std ≈ 1
   - But individual samples can be 2-3 std devs away (values of ±3)
   - Outliers can be 5-10 std devs away (values of ±10)
   - Extreme outliers can be even further (we saw ±50)

### 8. **What is the usual implementation in ML?**

**Standard practice** for regression tasks:

```python
# 1. During data generation/preprocessing:
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0) + 1e-8  # Add epsilon for stability
data_normalized = (data - data_mean) / data_std
save(data_normalized, data_mean, data_std)

# 2. During training:
batch = load_batch()  # Already normalized
predictions = model(batch['inputs'])  # Predicts normalized values
loss = MSE(predictions, batch['targets'])  # Both normalized

# 3. During inference/deployment:
input_norm = (input_raw - input_mean) / input_std
output_norm = model(input_norm)
output_raw = output_norm * output_std + output_mean  # Denormalize!
```

## What Was Wrong in Your Code

### Data Generation (`data_generation_jax.py`)
✅ **Correct**: Computed normalization stats
❌ **Wrong**: Stored RAW unnormalized data in HDF5

### Data Loader (`data_loader.py`)  
✅ **Correct**: Loaded normalization stats from HDF5
❌ **Wrong**: Never applied them to the data!

### Training (`train_jax.py`)
✅ **Correct**: Loss calculation, gradient clipping, optimizer
❌ **Victim**: Received unnormalized data, couldn't learn

## The Fix

**Modified `data_loader.py`** to normalize data on-the-fly:

```python
# In __next__() method, after loading from HDF5:

# Normalize states in input sequences
state_dim = self.state_mean.shape[0]
input_seq_states = input_seq[:, :, :state_dim]
input_seq_encoding = input_seq[:, :, state_dim:]
input_seq_states_norm = (input_seq_states - self.state_mean) / self.state_std
input_seq = np.concatenate([input_seq_states_norm, input_seq_encoding], axis=-1)

# Normalize controls (targets)
control = (control - self.control_mean) / self.control_std
```

## Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Loss (Batch 1)** | 15,781 | 8.8 | **1,792x** |
| **Target std** | 67.8 | 1.5 | **45x** |
| **Target range** | -2686 to 1793 | -15 to 54 | **88x** |
| **Max sample loss** | 7,217,821 | 1,186 | **6,087x** |
| **Gradient clipping** | 111x | 2.9x | **38x less severe** |

## Next Steps

1. ✅ **DONE**: Fixed normalization in data loader
2. 📊 **RECOMMENDED**: Train on Great Lakes GPU (3-4 hours vs 2 days on CPU)
3. 📈 **EXPECTED**: Loss should drop from ~8 → ~0.5-2.0 over 20-50 epochs
4. 🎯 **TARGET**: Final loss < 0.5 for deployment

## Great Lakes Training

See `GREAT_LAKES_SETUP.md` and `train_gpu.slurm` for:
- JAX GPU installation on Great Lakes
- Recommended cluster configuration (1x V100, 32GB RAM)
- Estimated training time: **3-4 hours** (vs 2 days on M4 CPU)
- Expected speedup: **30-60x faster**

---

**Status**: 🟢 **READY TO TRAIN** with properly normalized data!

