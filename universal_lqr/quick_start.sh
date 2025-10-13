#!/bin/bash
# Quick Start Script for JAX Implementation

set -e

echo "================================================================================"
echo "                  UNIVERSAL LQR TRANSFORMER - JAX IMPLEMENTATION"
echo "================================================================================"
echo ""

# Check if JAX is installed
if python -c "import jax" 2>/dev/null; then
    echo "✓ JAX is installed"
    python -c "import jax; print(f'  Backend: {jax.default_backend()}'); print(f'  Devices: {jax.devices()}')"
else
    echo "✗ JAX not found. Installing..."
    pip install jax jaxlib
fi

echo ""
echo "================================================================================"
echo "STEP 1: Test JAX Systems"
echo "================================================================================"
echo ""

python test_jax_systems.py

echo ""
echo "================================================================================"
echo "STEP 2: Generate Training Data"
echo "================================================================================"
echo ""
echo "Choose data generation method:"
echo "  1) JAX-accelerated (10-50x faster, recommended)"
echo "  2) Standard (compatible, slower)"
echo ""
read -p "Enter choice [1-2]: " data_choice

case $data_choice in
    1)
        echo "Using JAX-accelerated data generation..."
        python data_generation_jax.py
        ;;
    2)
        echo "Using standard data generation..."
        cd ..
        python data_generation.py
        cd jax_implementation
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "STEP 3: Train Transformer Model"
echo "================================================================================"
echo ""

python train_jax.py

echo ""
echo "================================================================================"
echo "✓ QUICK START COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Check trained model in: ../models/"
echo "  2. View training logs in: ../logs/"
echo "  3. Evaluate model: python ../evaluate.py"
echo ""
echo "For more information, see README.md"

