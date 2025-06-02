import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from data.prepare_data import load_processed_data, create_chord_mapper_from_metadata

class ChordMelodyDataset(Dataset):
    """Dataset for chord-to-melody generation"""
    
    def __init__(self, split='train', data_dir='data/processed'):
        self.split = split
        self.data_dir = Path(data_dir)
        
        # Load data
        train_data, test_data, metadata = load_processed_data()
        
        if split == 'train':
            self.data = train_data
        elif split == 'test':
            self.data = test_data
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.chord_sequences = self.data['chord_sequences']  # [N]
        self.melody_sequences = self.data['melody_sequences']  # [N, 32]
        self.metadata = metadata
        
        # Create chord mapper from metadata
        self.chord_mapper = create_chord_mapper_from_metadata(metadata)
        
        print(f"Loaded {split} dataset:")
        print(f"  🎵 Chord sequences: {self.chord_sequences.shape}")
        print(f"  🎼 Melody sequences: {self.melody_sequences.shape}")
        print(f"  🎹 Vocabulary size (notes): 128")
        print(f"  🎶 Chord types: {self.chord_mapper.num_chords}")
    
    def __len__(self):
        return len(self.chord_sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            chord_id: int - chord ID for the current bar
            melody: tensor [32] - melody sequence for the bar
        """
        chord_id = self.chord_sequences[idx]
        melody = self.melody_sequences[idx]
        
        return {
            'chord_id': chord_id,
            'melody': melody
        }
    
    def get_chord_name(self, chord_id):
        """Convert chord ID to chord name"""
        return self.chord_mapper.id_to_chord[chord_id.item() if torch.is_tensor(chord_id) else chord_id]
    
    def get_stats(self):
        """Get dataset statistics"""
        unique_chords, counts = torch.unique(self.chord_sequences, return_counts=True)
        chord_distribution = {}
        for chord_id, count in zip(unique_chords, counts):
            chord_name = self.get_chord_name(chord_id)
            chord_distribution[chord_name] = count.item()
        
        return {
            'num_samples': len(self),
            'chord_distribution': chord_distribution,
            'avg_confidence': self.metadata['stats']['avg_chord_confidence']
        }

def create_data_loaders(batch_size=32, data_dir='data/processed'):
    """Create train and test data loaders"""
    
    train_dataset = ChordMelodyDataset('train', data_dir)
    test_dataset = ChordMelodyDataset('test', data_dir)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader, train_dataset.metadata

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    chord_ids = torch.stack([item['chord_id'] for item in batch])
    melodies = torch.stack([item['melody'] for item in batch])
    
    return {
        'chord_ids': chord_ids,  # [batch_size]
        'melodies': melodies     # [batch_size, 32]
    }

if __name__ == "__main__":
    # Test the dataset
    print("🧪 Testing ChordMelodyDataset...")
    
    try:
        # Create datasets
        train_dataset = ChordMelodyDataset('train')
        test_dataset = ChordMelodyDataset('test')
        
        # Print statistics
        print("\n📊 Train Dataset Stats:")
        train_stats = train_dataset.get_stats()
        print(f"  Samples: {train_stats['num_samples']}")
        print(f"  Chord distribution: {train_stats['chord_distribution']}")
        
        print("\n📊 Test Dataset Stats:")
        test_stats = test_dataset.get_stats()
        print(f"  Samples: {test_stats['num_samples']}")
        
        # Test data loading
        print("\n🔄 Testing data loading...")
        train_loader, test_loader, metadata = create_data_loaders(batch_size=8)
        
        # Get a sample batch
        for batch in train_loader:
            chord_ids = batch['chord_id']
            melodies = batch['melody']
            
            print(f"  Batch chord IDs shape: {chord_ids.shape}")
            print(f"  Batch melodies shape: {melodies.shape}")
            
            # Show sample chord names
            sample_chords = [train_dataset.get_chord_name(cid) for cid in chord_ids[:5]]
            print(f"  Sample chords: {sample_chords}")
            
            break
        
        print("\n✅ Dataset test completed successfully!")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc() 