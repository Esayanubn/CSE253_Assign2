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

class MusicDataset(Dataset):
    def __init__(self, data_path, sequence_length=32, split='train'):
        self.sequence_length = sequence_length
        data = torch.load(data_path)
        
        # Split training and validation sets
        chord_sequences = data['chord_sequences']
        melody_sequences = data['melody_sequences']
        
        # Ensure sequence lengths are consistent
        valid_pairs = []
        for chord_seq, melody_seq in zip(chord_sequences, melody_sequences):
            if len(chord_seq) == len(melody_seq) and len(chord_seq) >= sequence_length:
                valid_pairs.append((chord_seq, melody_seq))
        
        if not valid_pairs:
            raise ValueError("No valid training data found! Please run the data processing script first.")
        
        # Randomly split data
        train_pairs, val_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
        
        if split == 'train':
            self.pairs = train_pairs
        else:
            self.pairs = val_pairs
        
        print(f"{split} set size: {len(self.pairs)}")
        print(f"Average sequence length: {np.mean([len(seq[0]) for seq in self.pairs]):.2f}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        chord_sequence, melody_sequence = self.pairs[idx]
        
        # Randomly select starting position
        if len(chord_sequence) > self.sequence_length:
            start = np.random.randint(0, len(chord_sequence) - self.sequence_length)
            chord_sequence = chord_sequence[start:start + self.sequence_length]
            melody_sequence = melody_sequence[start:start + self.sequence_length]
        
        return torch.tensor(chord_sequence), torch.tensor(melody_sequence)

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    best_val_loss = float('inf')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (chord_sequence, melody_sequence) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            chord_sequence = chord_sequence.to(device)
            melody_sequence = melody_sequence.to(device)
            
            optimizer.zero_grad()
            output = model(chord_sequence, melody_sequence)
            
            # Calculate loss for each time step
            loss = criterion(output.view(-1, output.size(-1)), melody_sequence.view(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for chord_sequence, melody_sequence in val_loader:
                chord_sequence = chord_sequence.to(device)
                melody_sequence = melody_sequence.to(device)
                
                output = model(chord_sequence, melody_sequence)
                loss = criterion(output.view(-1, output.size(-1)), melody_sequence.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
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
        if (epoch + 1) % 5 == 0:
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
    BATCH_SIZE = 16
    SEQUENCE_LENGTH = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0005
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    
    # Create datasets and data loaders
    train_dataset = MusicDataset('data/processed/processed_data.pt', SEQUENCE_LENGTH, split='train')
    val_dataset = MusicDataset('data/processed/processed_data.pt', SEQUENCE_LENGTH, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    model = ChordToMelodyTransformer(vocab_size=128, batch_first=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, DEVICE)

if __name__ == "__main__":
    main() 