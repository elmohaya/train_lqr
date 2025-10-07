"""
Diagnostic script to identify training bottlenecks
Run this on your server to check GPU utilization and data loading speed
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import numpy as np
import os
import time
from tqdm import tqdm

from config import TRAINING_CONFIG, DATA_DIR, MAX_STATE_DIM, MAX_INPUT_DIM, DIMENSION_ENCODING_SIZE
from train_fast import FastH5Dataset, UniversalLQRTransformer
from transformer_model import count_parameters

print("="*70)
print("TRAINING SPEED DIAGNOSTICS")
print("="*70)

# Check GPUs
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"\nGPUs detected: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda')
else:
    print("\nNo CUDA GPUs detected!")
    exit(1)

# Load data
h5_file = os.path.join(DATA_DIR, 'training_data.h5')
print(f"\nLoading data from: {h5_file}")

with h5py.File(h5_file, 'r') as hf:
    total_sequences = hf['input_sequences'].shape[0]
    print(f"Total sequences: {total_sequences:,}")

# Create small dataset for testing
test_size = 10000
indices = np.arange(test_size)
dataset = FastH5Dataset(h5_file, indices)

# Test 1: Data loading speed
print("\n" + "="*70)
print("TEST 1: Data Loading Speed")
print("="*70)

for num_workers in [0, 4, 8, 16, 24, 32]:
    print(f"\nTesting with {num_workers} workers...")
    
    loader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    start_time = time.time()
    n_batches = 0
    for batch in loader:
        n_batches += 1
        if n_batches >= 10:  # Test first 10 batches
            break
    elapsed = time.time() - start_time
    
    batches_per_sec = n_batches / elapsed
    samples_per_sec = (n_batches * TRAINING_CONFIG['batch_size']) / elapsed
    
    print(f"  Time for 10 batches: {elapsed:.2f}s")
    print(f"  Speed: {batches_per_sec:.1f} batches/sec, {samples_per_sec:.0f} samples/sec")

# Test 2: GPU computation speed
print("\n" + "="*70)
print("TEST 2: GPU Computation Speed")
print("="*70)

# Create model
from transformer_model import UniversalLQRTransformer
model = UniversalLQRTransformer(
    max_state_dim=MAX_STATE_DIM,
    max_control_dim=MAX_INPUT_DIM,
    dimension_encoding_size=DIMENSION_ENCODING_SIZE,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    dropout=0.1,
    max_seq_len=128
)

print(f"\nModel parameters: {count_parameters(model):,}")

# Test single GPU
print("\nSingle GPU:")
model_single = model.to(device)

dummy_input = torch.randn(TRAINING_CONFIG['batch_size'], 32, MAX_STATE_DIM + DIMENSION_ENCODING_SIZE).to(device)
dummy_mask = torch.ones(TRAINING_CONFIG['batch_size'], MAX_INPUT_DIM).to(device)

# Warmup
for _ in range(5):
    with torch.amp.autocast('cuda'):
        output = model_single(dummy_input, control_mask=dummy_mask)

# Benchmark
torch.cuda.synchronize()
start_time = time.time()
n_iterations = 50
for _ in range(n_iterations):
    with torch.amp.autocast('cuda'):
        output = model_single(dummy_input, control_mask=dummy_mask)
        loss = output.mean()
    loss.backward()
torch.cuda.synchronize()
elapsed = time.time() - start_time

single_gpu_speed = n_iterations / elapsed
print(f"  Forward+Backward: {single_gpu_speed:.1f} iterations/sec")
print(f"  Time per batch: {1000*elapsed/n_iterations:.1f}ms")

# Test multi-GPU
if n_gpus > 1:
    print(f"\nMulti-GPU ({n_gpus} GPUs with DataParallel):")
    model_multi = nn.DataParallel(model).to(device)
    
    # Warmup
    for _ in range(5):
        with torch.amp.autocast('cuda'):
            output = model_multi(dummy_input, control_mask=dummy_mask)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_iterations):
        with torch.amp.autocast('cuda'):
            output = model_multi(dummy_input, control_mask=dummy_mask)
            loss = output.mean()
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    multi_gpu_speed = n_iterations / elapsed
    speedup = multi_gpu_speed / single_gpu_speed
    
    print(f"  Forward+Backward: {multi_gpu_speed:.1f} iterations/sec")
    print(f"  Time per batch: {1000*elapsed/n_iterations:.1f}ms")
    print(f"  Speedup vs single GPU: {speedup:.2f}x")
    print(f"  Efficiency: {100*speedup/n_gpus:.1f}% (ideal = 100%)")

# Test 3: End-to-end training loop
print("\n" + "="*70)
print("TEST 3: End-to-End Training Loop")
print("="*70)

# Optimal configuration
num_workers = n_gpus * 8
loader = DataLoader(
    dataset,
    batch_size=TRAINING_CONFIG['batch_size'],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)

model = UniversalLQRTransformer(
    max_state_dim=MAX_STATE_DIM,
    max_control_dim=MAX_INPUT_DIM,
    dimension_encoding_size=DIMENSION_ENCODING_SIZE,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    dropout=0.1,
    max_seq_len=128
)

if n_gpus > 1:
    model = nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
scaler = torch.amp.GradScaler('cuda')

print(f"\nRunning 20 training iterations with {num_workers} workers...")
model.train()
start_time = time.time()
n_iterations = 0

for batch in tqdm(loader, desc="Training", total=20):
    input_sequence = batch['input_sequence'].to(device, non_blocking=True)
    controls_target = batch['control'].to(device, non_blocking=True)
    control_mask = batch['control_mask'].to(device, non_blocking=True)
    
    optimizer.zero_grad()
    
    with torch.amp.autocast('cuda'):
        controls_pred = model(input_sequence, control_mask=control_mask)
        controls_pred_last = controls_pred[:, -1, :]
        
        loss_per_dim = (controls_pred_last - controls_target) ** 2
        masked_loss = loss_per_dim * control_mask
        loss = masked_loss.sum() / control_mask.sum()
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    n_iterations += 1
    if n_iterations >= 20:
        break

elapsed = time.time() - start_time
batches_per_sec = n_iterations / elapsed
samples_per_sec = (n_iterations * TRAINING_CONFIG['batch_size']) / elapsed

print(f"\nResults:")
print(f"  Time for 20 batches: {elapsed:.2f}s")
print(f"  Speed: {batches_per_sec:.2f} batches/sec")
print(f"  Throughput: {samples_per_sec:.0f} samples/sec")
print(f"  Time per batch: {1000*elapsed/n_iterations:.1f}ms")

# Estimate epoch time
total_batches = total_sequences // TRAINING_CONFIG['batch_size']
estimated_epoch_time = total_batches / batches_per_sec / 60  # minutes

print(f"\nEstimated time per epoch:")
print(f"  Total batches: {total_batches:,}")
print(f"  Time: {estimated_epoch_time:.1f} minutes ({estimated_epoch_time/60:.2f} hours)")
print(f"  Total training (30 epochs): {30*estimated_epoch_time/60:.1f} hours")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if batches_per_sec < 5:
    print("\n[WARNING] Training is SLOW!")
    print("Possible issues:")
    print("  1. Data loading bottleneck - increase num_workers")
    print("  2. Small batch size - increase to 2048 or 4096")
    print("  3. Slow disk I/O - copy HDF5 file to /dev/shm (RAM disk)")
    print("  4. CPU bottleneck - check CPU usage with 'htop'")
elif batches_per_sec < 15:
    print("\n[OK] Training speed is acceptable but could be better.")
    print("Try:")
    print("  1. Increase SEQUENCE_STRIDE to 32 in config.py")
    print("  2. Reduce model size if not needed")
else:
    print("\n[EXCELLENT] Training speed is optimal!")
    print(f"Your setup should complete 30 epochs in ~{30*estimated_epoch_time/60:.1f} hours")

print("\n" + "="*70)

