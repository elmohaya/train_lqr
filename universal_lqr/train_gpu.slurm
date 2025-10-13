#!/bin/bash
#SBATCH --job-name=ulqr_transformer
#SBATCH --account=YOUR_ACCOUNT          # REPLACE WITH YOUR GREAT LAKES ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gpus-per-node=v100:1          # Options: v100:1, a100:1, a40:1
#SBATCH --mem=32GB                      # 32GB RAM (reduce to 16GB if using fast model)
#SBATCH --cpus-per-task=4               # 4 CPU cores for data loading
#SBATCH --time=24:00:00                 # 24 hours (should finish in 3-4h on V100)
#SBATCH --output=logs/train_%j.out      # Standard output
#SBATCH --error=logs/train_%j.err       # Standard error
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=YOUR_EMAIL@umich.edu  # REPLACE WITH YOUR EMAIL

# ============================================================================
# Great Lakes GPU Training Script for Universal LQR Transformer
# ============================================================================

echo "========================================================================"
echo "=== Universal LQR Transformer Training on Great Lakes GPU ============="
echo "========================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

# Load required modules
echo "Loading modules..."
module load cuda/11.8
module load python/3.10
module list

# Activate virtual environment
echo "Activating Python environment..."
source /home/YOUR_UNIQNAME/universal_lqr_training/jax_gpu_env/bin/activate  # REPLACE YOUR_UNIQNAME

# Create logs and models directories if they don't exist
mkdir -p logs
mkdir -p models

# Verify GPU availability
echo ""
echo "========================================================================"
echo "=== GPU Verification =================================================="
echo "========================================================================"
nvidia-smi
echo ""
python -c "import jax; print('JAX version:', jax.__version__); print('Available devices:', jax.devices()); print('GPU available:', jax.devices()[0].platform == 'gpu')"
echo "========================================================================"
echo ""

# Navigate to project directory
cd /home/YOUR_UNIQNAME/universal_lqr_training/jax_implementation  # REPLACE YOUR_UNIQNAME

# Check if data file exists
if [ ! -f "data/training_data_jax.h5" ]; then
    echo "ERROR: Data file not found at data/training_data_jax.h5"
    echo "Please transfer your data file first!"
    exit 1
fi

echo "Data file found: data/training_data_jax.h5"
echo ""

# ============================================================================
# Choose training configuration
# ============================================================================

# OPTION A: Full model (208k params, 50 epochs) - Recommended for final results
echo "========================================================================"
echo "=== Starting Training with FULL MODEL (208k parameters) =============="
echo "========================================================================"
python train_jax.py

# OPTION B: Fast model (30k params, 20 epochs) - Uncomment below for quick testing
# echo "========================================================================"
# echo "=== Starting Training with FAST MODEL (30k parameters) ==============="
# echo "========================================================================"
# # Modify train_jax.py to import config_fast_cpu instead of config
# python train_jax.py

# ============================================================================
# Job completion
# ============================================================================

echo ""
echo "========================================================================"
echo "=== Training Complete ================================================="
echo "========================================================================"
echo "Job finished at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================================================"
echo "Trained models saved in: ./models/"
echo "Training logs saved in: ./logs/"
echo "========================================================================"

# Show final GPU memory usage
nvidia-smi

exit 0

