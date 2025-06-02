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

from models.transformer import ChordToMelodyTransformer
from utils.midi_utils import save_training_samples

class MusicDataset(Dataset):
    def __init__(self, split='train', sequence_length=32):
        """
        Initialize dataset with new data format
        Args:
            split: 'train' or 'test'
            sequence_length: length of sequences (should be 32 based on our data)
        """
        self.sequence_length = sequence_length
        
        # Load the appropriate data file
        data_path = f"data/processed/{split}_data.pt"
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        data = torch.load(data_path, weights_only=True)
        
        # Extract chord and melody sequences
        self.chord_sequences = data['chord_sequences']  # Shape: [N] - single chord ID per sequence
        self.melody_sequences = data['melody_sequences']  # Shape: [N, 32] - melody sequences
        
        print(f"Loading {split} dataset:")
        print(f"  Chord sequences shape: {self.chord_sequences.shape}")
        print(f"  Melody sequences shape: {self.melody_sequences.shape}")
        print(f"  Chord sequences dtype: {self.chord_sequences.dtype}")
        print(f"  Melody sequences dtype: {self.melody_sequences.dtype}")
        print(f"  Number of sequences: {len(self.chord_sequences)}")
        
        # Validate data
        assert len(self.chord_sequences) == len(self.melody_sequences), "Chord and melody sequences must have same length"
        assert self.melody_sequences.shape[1] == sequence_length, f"Melody sequences must have length {sequence_length}"
        
        print(f"  Chord ID range: [{self.chord_sequences.min()}, {self.chord_sequences.max()}]")
        print(f"  Melody note range: [{self.melody_sequences.min()}, {self.melody_sequences.max()}]")
    
    def __len__(self):
        return len(self.chord_sequences)
    
    def __getitem__(self, idx):
        """
        Get a single training sample
        Returns:
            chord_id: single chord ID for the entire sequence
            melody_sequence: sequence of melody notes [32]
        """
        chord_id = self.chord_sequences[idx]  # Single chord ID
        melody_sequence = self.melody_sequences[idx]  # [32] melody notes
        
        # Ensure proper data types
        chord_id = chord_id.long()
        melody_sequence = melody_sequence.long()
        
        return chord_id, melody_sequence

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
        
        for batch_idx, (chord_ids, melody_sequences) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            try:
                chord_ids = chord_ids.to(device).long()  # [batch_size] - single chord per sequence
                melody_sequences = melody_sequences.to(device).long()  # [batch_size, 32]
                
                if batch_idx % 50 == 0:  # Print every 50 batches
                    print(f"Batch {batch_idx}: chord shape {chord_ids.shape}, melody shape {melody_sequences.shape}")
                    print(f"  Chord ID range: [{chord_ids.min()}, {chord_ids.max()}]")
                    print(f"  Melody range: [{melody_sequences.min()}, {melody_sequences.max()}]")
                
                optimizer.zero_grad()
                
                # Use teacher forcing for training
                if melody_sequences.size(1) > 1:
                    # Input: melody[:-1], Target: melody[1:]
                    melody_input = melody_sequences[:, :-1].contiguous()  # [batch_size, 31]
                    melody_target = melody_sequences[:, 1:].contiguous()  # [batch_size, 31]
                    
                    # Pass chord_ids and melody_input to model
                    # chord_ids: [batch_size] - single chord conditioning each sequence
                    output = model(chord_ids, melody_input)  # Expected output: [batch_size, 31, vocab_size]
                    
                    # Flatten for loss calculation
                    output_flat = output.contiguous().view(-1, output.size(-1))  # [batch_size*31, vocab_size]
                    target_flat = melody_target.contiguous().view(-1)  # [batch_size*31]
                    
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
            for chord_ids, melody_sequences in val_loader:
                try:
                    chord_ids = chord_ids.to(device).long()
                    melody_sequences = melody_sequences.to(device).long()
                    
                    if melody_sequences.size(1) > 1:
                        melody_input = melody_sequences[:, :-1].contiguous()
                        melody_target = melody_sequences[:, 1:].contiguous()
                        
                        output = model(chord_ids, melody_input)
                        
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
        if (epoch + 1) % 5 == 0 and metadata is not None:
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
    # Set parameters
    BATCH_SIZE = 8  # Increased batch size since data loading is now simpler
    SEQUENCE_LENGTH = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("output/training_samples").mkdir(parents=True, exist_ok=True)
    
    # Load metadata to get vocabulary information
    try:
        metadata = torch.load('data/processed/metadata.pt', weights_only=True)
        num_chords = metadata['num_chords']
        print(f"Number of chord types: {num_chords}")
        print(f"Average chord confidence: {metadata['stats']['avg_chord_confidence']:.3f}")
    except FileNotFoundError:
        print("Warning: metadata.pt not found, using default values")
        num_chords = 49  # Default based on ChordMapper
        metadata = None
    
    # Create datasets and data loaders
    try:
        train_dataset = MusicDataset(split='train', sequence_length=SEQUENCE_LENGTH)
        val_dataset = MusicDataset(split='test', sequence_length=SEQUENCE_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Create model with appropriate vocab size
        # vocab_size=128 for MIDI notes, chord_vocab_size for chord conditioning
        model = ChordToMelodyTransformer(
            vocab_size=128, 
            chord_vocab_size=num_chords,  # Number of chord types
            d_model=256, 
            nhead=8
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