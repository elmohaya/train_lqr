#!/bin/bash
# Script to install JAX with CUDA support on Linux server

echo "========================================"
echo "Installing JAX with CUDA Support"
echo "========================================"

# Check CUDA version
echo ""
echo "Step 1: Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "WARNING: nvcc not found. Checking nvidia-smi..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo ""
        echo "Please check the CUDA Version in the output above."
        echo "Common versions: 11.8, 12.0, 12.1, 12.2, 12.3"
        read -p "Enter your CUDA version (e.g., 12.1): " CUDA_VERSION
    else
        echo "ERROR: Cannot detect CUDA. Please install CUDA first."
        exit 1
    fi
fi

# Uninstall existing JAX
echo ""
echo "Step 2: Uninstalling existing JAX (if any)..."
pip3 uninstall -y jax jaxlib

# Install JAX with CUDA support
echo ""
echo "Step 3: Installing JAX with CUDA $CUDA_VERSION support..."

# Determine CUDA version for JAX installation
if [[ "$CUDA_VERSION" == 11.* ]]; then
    echo "Installing JAX for CUDA 11..."
    pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
elif [[ "$CUDA_VERSION" == 12.* ]]; then
    echo "Installing JAX for CUDA 12..."
    pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "WARNING: Unknown CUDA version. Trying CUDA 12 by default..."
    pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# Install other dependencies
echo ""
echo "Step 4: Installing other dependencies..."
pip3 install --upgrade flax optax

# Verify installation
echo ""
echo "========================================"
echo "Step 5: Verifying JAX installation..."
echo "========================================"
python3 -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Device count:', jax.device_count())
print('')
if jax.device_count() > 0 and 'gpu' in str(jax.devices()[0]).lower():
    print('[OK] JAX successfully installed with GPU support!')
    print('GPUs detected:', jax.device_count())
else:
    print('[WARNING] JAX installed but not detecting GPUs.')
    print('Please check your CUDA installation.')
"

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"

