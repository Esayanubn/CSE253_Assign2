#!/usr/bin/env python3
"""
Quick training script to demonstrate MIDI generation functionality
"""
import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from train import MusicDataset, train_model
from models.transformer import ChordToMelodyTransformer

def quick_train():
    """Run a quick training session with MIDI generation"""
    print("🎼 Quick Training with MIDI Generation Demo")
    print("=" * 50)
    
    # Set parameters for quick demo
    BATCH_SIZE = 8
    SEQUENCE_LENGTH = 32
    NUM_EPOCHS = 10  # Only 10 epochs for demo
    LEARNING_RATE = 0.0001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Training for {NUM_EPOCHS} epochs")
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("output/training_samples").mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    try:
        metadata = torch.load('data/processed/metadata.pt', weights_only=True)
        num_chords = metadata['num_chords']
        print(f"Loaded {num_chords} chord types")
    except FileNotFoundError:
        print("Error: metadata.pt not found")
        return
    
    # Create datasets
    try:
        train_dataset = MusicDataset(split='train', sequence_length=SEQUENCE_LENGTH)
        val_dataset = MusicDataset(split='test', sequence_length=SEQUENCE_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Create model
        model = ChordToMelodyTransformer(
            vocab_size=128, 
            chord_vocab_size=num_chords,
            d_model=256, 
            nhead=8
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model - will generate MIDI every 5 epochs (epochs 5 and 10)
        train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, DEVICE, metadata)
        
        print("\n🎵 Training complete! Check output/training_samples/ for generated MIDI files")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_train() 