# Installing JAX on macOS (Apple Silicon M1/M2/M3/M4)

## For MacBook Pro with M4 (Apple Silicon)

### Option 1: JAX with Metal (GPU Support) ⭐ Recommended

JAX now supports Apple's Metal backend for GPU acceleration on M-series chips.

```bash
# Install JAX with Metal support
pip install jax-metal

# This will also install jax and jaxlib
```

**Check installation:**
```bash
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
```

You should see:
```
JAX version: 0.4.x
Devices: [METAL(id=0)]
```

### Option 2: JAX CPU-only (Fallback)

If Metal has issues, use CPU-only version:

```bash
pip install jax
```

**Check installation:**
```bash
python -c "import jax; print('Devices:', jax.devices())"
```

You should see:
```
Devices: [CpuDevice(id=0)]
```

## Performance Expectations

### M4 MacBook Pro Performance

| Mode | Data Generation (100M) | Training | Notes |
|------|------------------------|----------|-------|
| **Metal (GPU)** | ~2-4 hours | Fast | ⭐ Best for M4 |
| **CPU only** | ~4-8 hours | Medium | Still faster than NumPy |
| **No JAX (NumPy)** | ~12-24 hours | Slow | Not recommended |

### Why Metal is Good for M4:
- ✅ Unified memory (shared RAM/GPU)
- ✅ High bandwidth (~400GB/s on M4 Pro/Max)
- ✅ Energy efficient
- ✅ 10-20x faster than CPU
- ⚠️ Not as fast as NVIDIA GPUs (but very good!)

## Quick Test

After installing JAX:

```bash
cd jax_implementation

# Test JAX installation
python -c "
import jax
import jax.numpy as jnp
print('✓ JAX installed successfully')
print(f'Backend: {jax.default_backend()}')
print(f'Devices: {jax.devices()}')

# Quick performance test
x = jnp.ones((1000, 1000))
result = jnp.dot(x, x)
print(f'✓ Matrix multiplication works: {result.shape}')
"

# Test systems
python test_jax_systems.py

# Generate small dataset (test)
python data_generation_jax.py  # Will ask for confirmation
```

## Troubleshooting

### Issue: "jax-metal not found"

```bash
# Update pip first
pip install --upgrade pip

# Try again
pip install jax-metal
```

### Issue: Metal not working

```bash
# Check Metal availability
python -c "
import jax
print('Available backends:', jax.lib.xla_bridge.get_backend().platform)
"

# If Metal fails, fall back to CPU
pip uninstall jax-metal
pip install jax  # CPU version
```

### Issue: Memory errors

```bash
# Limit memory usage
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# Or reduce batch size in config.py
```

### Issue: Slow compilation (first run)

This is normal! JAX compiles functions on first run:
- First trajectory: ~10-30 seconds (compilation)
- Subsequent trajectories: <0.1 seconds (cached)

## Recommendations for M4 MacBook Pro

### For Data Generation:
```bash
# Install with Metal
pip install jax-metal

# Generate data (will use Metal GPU)
cd jax_implementation
python data_generation_jax.py
```

### For Training:
```bash
# Same - Metal GPU is good for training
python train_jax.py
```

### If You Have Issues:
```bash
# Use CPU version (still faster than NumPy)
pip install jax
cd jax_implementation
python data_generation_jax.py
```

## M4-Specific Notes

The M4 chip has:
- **Powerful Neural Engine** - JAX can leverage this via Metal
- **High memory bandwidth** - Great for large batches
- **Unified memory** - Efficient CPU↔GPU transfer

**Bottom line:** JAX with Metal on M4 should work very well! 🚀

## Alternative: Use NumPy Version

If JAX installation fails, you can still use the main (NumPy) implementation:

```bash
cd ..  # Go back to main directory
python generate_fast_data.py  # NumPy implementation (slower but works)
```

## References

- JAX on Apple Silicon: https://github.com/google/jax/issues/5501
- JAX Metal backend: https://developer.apple.com/metal/jax/
- JAX installation: https://github.com/google/jax#installation
