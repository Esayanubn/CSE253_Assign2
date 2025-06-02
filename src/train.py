import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer import ChordToMelodyTransformer
from utils.midi_utils import save_training_samples
from config import *

class MusicDataset(Dataset):
    def __init__(self, split='train'):
        """
        Initialize dataset with new long sequence data format
        Args:
            split: 'train' or 'test'
        """
        
        # Load the appropriate data file (using long sequence data)
        data_path = f"data/processed/{split}_data_long.pt"
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        data = torch.load(data_path, weights_only=True)
        
        # Extract chord and melody sequences
        self.chord_sequences = data['chord_sequences']  # Shape: [N, 512] - chord sequence per time step
        self.melody_sequences = data['melody_sequences']  # Shape: [N, 512] - melody sequences
        
        print(f"Loading {split} dataset:")
        print(f"  Chord sequences shape: {self.chord_sequences.shape}")
        print(f"  Melody sequences shape: {self.melody_sequences.shape}")
        print(f"  Chord sequences dtype: {self.chord_sequences.dtype}")
        print(f"  Melody sequences dtype: {self.melody_sequences.dtype}")
        print(f"  Number of sequences: {len(self.chord_sequences)}")
        
        # Validate data
        assert len(self.chord_sequences) == len(self.melody_sequences), "Chord and melody sequences must have same length"
        assert self.chord_sequences.shape[1] == SEQUENCE_LENGTH, f"Chord sequences must have length {SEQUENCE_LENGTH}"
        assert self.melody_sequences.shape[1] == SEQUENCE_LENGTH, f"Melody sequences must have length {SEQUENCE_LENGTH}"
        
        print(f"  Chord ID range: [{self.chord_sequences.min()}, {self.chord_sequences.max()}]")
        print(f"  Melody note range: [{self.melody_sequences.min()}, {self.melody_sequences.max()}]")
    
    def __len__(self):
        return len(self.chord_sequences)
    
    def __getitem__(self, idx):
        """
        Get a single training sample
        Returns:
            chord_sequence: chord sequence [512] - time-aligned with melody
            melody_sequence: sequence of melody notes [512]
        """
        chord_sequence = self.chord_sequences[idx]  # [512] chord sequence
        melody_sequence = self.melody_sequences[idx]  # [512] melody notes
        
        # Ensure proper data types
        chord_sequence = chord_sequence.long()
        melody_sequence = melody_sequence.long()
        
        return chord_sequence, melody_sequence

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, metadata=None):
    best_val_loss = float('inf')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    
    # Create output directory for MIDI samples
    output_dir = Path("output/training_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_idx, (chord_sequences, melody_sequences) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            try:
                chord_sequences = chord_sequences.to(device).long()  # [batch_size, 512] - chord sequence per time step
                melody_sequences = melody_sequences.to(device).long()  # [batch_size, 512] - melody sequences
                
                if batch_idx % 50 == 0:  # Print every 50 batches
                    print(f"Batch {batch_idx}: chord shape {chord_sequences.shape}, melody shape {melody_sequences.shape}")
                    print(f"  Chord ID range: [{chord_sequences.min()}, {chord_sequences.max()}]")
                    print(f"  Melody range: [{melody_sequences.min()}, {melody_sequences.max()}]")
                
                optimizer.zero_grad()
                
                # Use teacher forcing for training
                if melody_sequences.size(1) > 1:
                    # Input: melody[:-1], Target: melody[1:]
                    melody_input = melody_sequences[:, :-1].contiguous()  # [batch_size, 511]
                    melody_target = melody_sequences[:, 1:].contiguous()  # [batch_size, 511]
                    
                    # Also truncate chord sequences to match
                    chord_input = chord_sequences[:, :-1].contiguous()  # [batch_size, 511]
                    
                    # Pass chord_sequences and melody_input to model
                    # chord_input: [batch_size, 511] - chord sequence conditioning each time step
                    output = model(chord_input, melody_input)  # Expected output: [batch_size, 511, vocab_size]
                    
                    # Flatten for loss calculation
                    output_flat = output.contiguous().view(-1, output.size(-1))  # [batch_size*511, vocab_size]
                    target_flat = melody_target.contiguous().view(-1)  # [batch_size*511]
                    
                    # Calculate loss
                    loss = criterion(output_flat, target_flat)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_batches += 1
                
                if batch_idx % 50 == 0:
                    print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if num_batches == 0:
            print("No successful training batches, skipping epoch")
            continue
            
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for chord_sequences, melody_sequences in val_loader:
                try:
                    chord_sequences = chord_sequences.to(device).long()
                    melody_sequences = melody_sequences.to(device).long()
                    
                    if melody_sequences.size(1) > 1:
                        melody_input = melody_sequences[:, :-1].contiguous()
                        melody_target = melody_sequences[:, 1:].contiguous()
                        
                        # Also truncate chord sequences for validation
                        chord_input = chord_sequences[:, :-1].contiguous()
                        
                        output = model(chord_input, melody_input)
                        
                        # Flatten for loss calculation
                        output_flat = output.contiguous().view(-1, output.size(-1))
                        target_flat = melody_target.contiguous().view(-1)
                        
                        loss = criterion(output_flat, target_flat)
                        val_loss += loss.item()
                        num_val_batches += 1
                        
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if num_val_batches == 0:
            print("No successful validation batches")
            continue
            
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        # Generate MIDI samples every 5 epochs
        # if (epoch + 1) % 5 == 0 and metadata is not None:
        try:
            print(f"\n🎼 Generating MIDI samples at epoch {epoch+1}...")
            save_training_samples(
                model=model,
                train_loader=train_loader,
                metadata=metadata,
                device=device,
                output_dir=output_dir,
                epoch=epoch+1,
                num_samples=5
            )
        except Exception as e:
            print(f"Error generating MIDI samples: {e}")
            import traceback
            traceback.print_exc()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'models/best_model.pth')
            print("Best model saved!")
        
        # Periodic checkpoint saving
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'models/checkpoint_epoch_{epoch+1}.pt')
    
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('models/training_loss.png')
    plt.close()

def main():
    # Use config variables
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Configuration: {SEQUENCE_LENGTH} step sequences, batch size {BATCH_SIZE}")
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("output/training_samples").mkdir(parents=True, exist_ok=True)
    
    # Load metadata to get vocabulary information (use long sequence metadata)
    try:
        metadata = torch.load('data/processed/metadata_long.pt', weights_only=True)
        num_chords = metadata['num_chords']
        print(f"Number of chord types: {num_chords}")
        print(f"Average chord confidence: {metadata['stats']['avg_chord_confidence']:.3f}")
        print(f"Processed sequences: {metadata['stats']['processed_sequences']}")
        print(f"Train size: {metadata['stats']['train_size']}")
        print(f"Test size: {metadata['stats']['test_size']}")
    except FileNotFoundError:
        print("Warning: metadata_long.pt not found, using default values")
        num_chords = CHORD_VOCAB_SIZE  # Use config value
        metadata = None
    
    # Create datasets and data loaders
    try:
        train_dataset = MusicDataset(split='train')
        val_dataset = MusicDataset(split='test')
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Create model using config values
        model = ChordToMelodyTransformer(
            vocab_size=VOCAB_SIZE, 
            chord_vocab_size=num_chords,
            d_model=D_MODEL, 
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens (note 0)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model with metadata for MIDI generation
        train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, DEVICE, metadata)
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 