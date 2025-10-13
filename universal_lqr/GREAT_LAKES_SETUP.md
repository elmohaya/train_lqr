# 🖥️ Great Lakes GPU Training Setup Guide

## Overview
This guide shows how to train the Universal LQR Transformer on University of Michigan's Great Lakes cluster with GPU acceleration.

---

## 1. Check JAX GPU Support on Great Lakes

Great Lakes supports **NVIDIA GPUs with CUDA**, which is compatible with JAX!

### Available GPUs on Great Lakes:
- **V100 GPUs** (16GB/32GB VRAM) - Good for most tasks
- **A100 GPUs** (40GB VRAM) - Best performance
- **A40 GPUs** (48GB VRAM) - Excellent for large models

---

## 2. Recommended Cluster Configuration

### For This Project (Universal LQR Transformer):

#### **Option A: Standard Training (Full Model - 208k params)**
```bash
#SBATCH --job-name=universal_lqr
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gpus-per-node=v100:1    # or a100:1 for faster
#SBATCH --mem=32GB                # 32GB RAM (plenty for streaming data)
#SBATCH --cpus-per-task=4         # 4 CPU cores for data loading
#SBATCH --time=48:00:00           # 48 hours (should finish in ~10-20h on GPU)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@umich.edu
```

**Estimated Time:**
- **V100**: ~15-20 hours for 50 epochs
- **A100**: ~8-12 hours for 50 epochs

#### **Option B: Fast Training (Small Model - 30k params with config_fast_cpu.py)**
```bash
#SBATCH --job-name=universal_lqr_fast
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=16GB                # Less RAM needed
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00            # 6 hours (should finish in ~2-4h on GPU)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@umich.edu
```

**Estimated Time:**
- **V100**: ~3-4 hours for 20 epochs
- **A100**: ~1.5-2 hours for 20 epochs

---

## 3. Installation Steps on Great Lakes

### Step 1: Login and Setup
```bash
ssh YOUR_UNIQNAME@greatlakes.arc-ts.umich.edu
cd /home/YOUR_UNIQNAME
mkdir -p universal_lqr_training
cd universal_lqr_training
```

### Step 2: Load CUDA Module
```bash
# Check available CUDA versions
module avail cuda

# Load CUDA (use version 11.8 or 12.1 for JAX)
module load cuda/11.8

# Add to your .bashrc for persistent loading
echo "module load cuda/11.8" >> ~/.bashrc
```

### Step 3: Create Python Environment
```bash
# Load Anaconda/Python module
module load python/3.10

# Create virtual environment
python -m venv jax_gpu_env
source jax_gpu_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 4: Install JAX with GPU Support
```bash
# Install JAX with CUDA 11.8 support
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Or for CUDA 12:
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify JAX GPU installation
python -c "import jax; print('JAX devices:', jax.devices()); print('GPU available:', jax.devices()[0].platform == 'gpu')"
```

### Step 5: Install Other Requirements
```bash
# Install other dependencies
pip install flax optax h5py tqdm matplotlib

# Or use requirements file
pip install -r requirements_jax.txt
```

---

## 4. Transfer Data to Great Lakes

### Option A: Using SCP (from your local machine)
```bash
# Transfer the entire jax_implementation folder
scp -r /Users/turki/Desktop/My_PhD/highway_merging/ablation/universal_lqr/jax_implementation \
    YOUR_UNIQNAME@greatlakes.arc-ts.umich.edu:/home/YOUR_UNIQNAME/universal_lqr_training/

# Or just transfer the HDF5 data file (it's large!)
scp /Users/turki/Desktop/My_PhD/highway_merging/ablation/universal_lqr/jax_implementation/data/training_data_jax.h5 \
    YOUR_UNIQNAME@greatlakes.arc-ts.umich.edu:/home/YOUR_UNIQNAME/universal_lqr_training/jax_implementation/data/
```

### Option B: Using Globus (Recommended for large files)
Great Lakes supports Globus for large file transfers:
1. Go to https://www.globus.org/
2. Setup endpoint on your local machine
3. Connect to Great Lakes endpoint
4. Transfer files

---

## 5. Create SLURM Job Script

Create a file `train_gpu.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=ulqr_transformer
#SBATCH --account=YOUR_ACCOUNT          # REPLACE WITH YOUR ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gpus-per-node=v100:1          # or a100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@umich.edu

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment
source /home/YOUR_UNIQNAME/universal_lqr_training/jax_gpu_env/bin/activate

# Create logs directory if not exists
mkdir -p logs

# Print system info
echo "================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================"

# Verify GPU availability
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"

# Navigate to project directory
cd /home/YOUR_UNIQNAME/universal_lqr_training/jax_implementation

# Run training (FULL MODEL)
echo "Starting training with FULL model (208k params)..."
python train_jax.py

# Or run training with FAST model (30k params)
# echo "Starting training with FAST model (30k params)..."
# python train_jax.py --config config_fast_cpu

echo "================================"
echo "Job finished at: $(date)"
echo "================================"
```

Make it executable:
```bash
chmod +x train_gpu.slurm
```

---

## 6. Submit Job and Monitor

### Submit Job
```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch train_gpu.slurm

# Check job status
squeue -u YOUR_UNIQNAME

# Check detailed job info
scontrol show job JOB_ID
```

### Monitor Training
```bash
# Follow the output in real-time
tail -f logs/train_JOBID.out

# Check for errors
tail -f logs/train_JOBID.err

# Check GPU usage (from a login node after job starts)
ssh gl3001  # Replace with your job's node
nvidia-smi
```

### Cancel Job (if needed)
```bash
scancel JOB_ID
```

---

## 7. Modify train_jax.py for Optimal GPU Performance

Update `train_jax.py` to use larger batch size for GPU:

```python
# At the top of train_jax.py, detect GPU and adjust config
import jax

# Check if running on GPU
devices = jax.devices()
is_gpu = devices[0].platform == 'gpu'

if is_gpu:
    print("GPU detected! Using optimized batch size...")
    # Override batch size from config
    from config import *
    TRAINING_CONFIG['batch_size'] = 2048  # Larger batch for GPU
else:
    print("CPU detected! Using smaller batch size...")
    from config_fast_cpu import *
```

Or create a separate `config_gpu.py`:

```python
# config_gpu.py - Optimized for GPU training
from config import *

# Override for GPU performance
TRAINING_CONFIG['batch_size'] = 2048  # GPU can handle larger batches
TRAINING_CONFIG['num_epochs'] = 50   # Full training

print("GPU CONFIG LOADED: batch_size=2048, epochs=50")
```

---

## 8. Expected Performance on Great Lakes

### Training Speed Comparison:

| Hardware | Batch Size | Time/Epoch | Total Time (50 epochs) |
|----------|------------|------------|------------------------|
| **M4 CPU** | 256 | ~147 min | ~122 hours (~5 days) |
| **M4 CPU (fast)** | 512 | ~147 min | ~49 hours (~2 days) |
| **V100 GPU** | 2048 | ~3-5 min | **2.5-4 hours** ✅ |
| **A100 GPU** | 2048 | ~1-2 min | **1-2 hours** ✅✅ |

### GPU Speedup: **20-60x faster than CPU!**

---

## 9. Retrieve Results

After training completes:

```bash
# Download trained model
scp YOUR_UNIQNAME@greatlakes.arc-ts.umich.edu:/home/YOUR_UNIQNAME/universal_lqr_training/jax_implementation/models/best_model_jax.pkl \
    /Users/turki/Desktop/My_PhD/highway_merging/ablation/universal_lqr/jax_implementation/models/

# Download logs
scp YOUR_UNIQNAME@greatlakes.arc-ts.umich.edu:/home/YOUR_UNIQNAME/universal_lqr_training/logs/train_*.out \
    /Users/turki/Desktop/My_PhD/highway_merging/ablation/universal_lqr/logs/
```

---

## 10. Troubleshooting

### Issue: JAX doesn't detect GPU
```bash
# Check CUDA installation
nvcc --version

# Check if GPU is visible
nvidia-smi

# Reinstall JAX with correct CUDA version
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue: Out of Memory (OOM)
- **Reduce batch size** in config (try 1024 or 512)
- **Request more GPU memory** (use A100 with 40GB)
- **Use gradient accumulation** (modify train_jax.py)

### Issue: Job time limit exceeded
- **Request more time**: `--time=72:00:00`
- **Use faster GPU**: A100 instead of V100
- **Use smaller model**: config_fast_cpu.py

---

## 11. Cost/Priority Notes

Great Lakes uses **fairshare** allocation:
- **Standard GPU partition**: Free for all users (but may queue)
- **Priority GPU**: If you have specific allocations

Check your account:
```bash
my_accounts  # Shows your available accounts
```

---

## 12. Summary Checklist

- [ ] SSH into Great Lakes
- [ ] Load CUDA module (`module load cuda/11.8`)
- [ ] Create virtual environment
- [ ] Install JAX with GPU support
- [ ] Transfer code and data (use Globus for large files)
- [ ] Create SLURM job script
- [ ] Submit job (`sbatch train_gpu.slurm`)
- [ ] Monitor progress (`tail -f logs/train_*.out`)
- [ ] Retrieve trained model

---

## 🎯 Recommended Action for Your Case:

**Use V100 GPU with full model (208k params):**
- Request: 1x V100, 32GB RAM, 4 CPUs, 24 hours
- Estimated time: **~3-4 hours** (vs 5 days on M4 CPU!)
- Cost: Free (standard allocation)

This will give you **30x speedup** and finish overnight! 🚀

