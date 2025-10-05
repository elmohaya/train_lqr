"""
Main Execution Script for Universal LQR Transformer
"""

import argparse
import sys
import torch

from config import SAVE_NEW_DATA, DATA_DIR, MODEL_DIR
from data_generation import main as generate_data_main
from train import main as train_main


def main():
    parser = argparse.ArgumentParser(description='Universal LQR Transformer')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['data', 'train', 'all'],
                       help='Execution mode: data generation, training, or both')
    parser.add_argument('--force-data-gen', action='store_true',
                       help='Force data generation even if data exists')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" Universal LQR Transformer ".center(70, "="))
    print("="*70)
    print("\nGoal: Train a transformer to stabilize any unseen LTI system")
    print(f"\nMode: {args.mode}")
    print()
    
    # Data generation
    if args.mode in ['data', 'all']:
        print("\n" + "="*70)
        print(" STEP 1: DATA GENERATION ".center(70, "="))
        print("="*70 + "\n")
        
        if args.force_data_gen:
            print("Forcing new data generation...")
            import config
            config.SAVE_NEW_DATA = True
        
        generate_data_main()
    
    # Training
    if args.mode in ['train', 'all']:
        print("\n" + "="*70)
        print(" STEP 2: TRANSFORMER TRAINING ".center(70, "="))
        print("="*70 + "\n")
        
        train_main()
    
    print("\n" + "="*70)
    print(" COMPLETE ".center(70, "="))
    print("="*70 + "\n")
    print("Next steps:")
    print("  1. Evaluate model performance on test systems")
    print("  2. Test on completely new unseen LTI systems")
    print("  3. Analyze generalization capabilities")
    print()


if __name__ == '__main__':
    main()

