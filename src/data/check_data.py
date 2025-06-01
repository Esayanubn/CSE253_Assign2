import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_processed_data():
    """Check processed data"""
    data_path = Path("data/processed/processed_data.pt")
    if not data_path.exists():
        print("Error: Processed data file not found!")
        return
    
    # Load data
    data = torch.load(data_path)
    chord_sequences = data['chord_sequences']
    melody_sequences = data['melody_sequences']
    
    print(f"Dataset Statistics:")
    print(f"Total sequences: {len(chord_sequences)}")
    print(f"Average sequence length: {np.mean([len(seq) for seq in chord_sequences]):.2f}")
    print(f"Maximum sequence length: {max(len(seq) for seq in chord_sequences)}")
    print(f"Minimum sequence length: {min(len(seq) for seq in chord_sequences)}")
    
    # Check chord type distribution
    chord_types = []
    chord_roots = []
    for seq in chord_sequences:
        for chord in seq:
            chord_type = chord % 10
            chord_root = chord // 10
            # 只统计有效的根音（0-11）
            if 0 <= chord_root < 12:
                chord_types.append(chord_type)
                chord_roots.append(chord_root)
    
    # Chord type statistics
    chord_type_counts = np.bincount(chord_types)
    chord_type_names = ['Other', 'Major', 'Minor', 'Major7', 'Minor7', 'Dominant7']
    
    print("\nChord Type Distribution:")
    total_chords = sum(chord_type_counts)
    for i, count in enumerate(chord_type_counts):
        if i < len(chord_type_names):
            percentage = (count / total_chords) * 100
            print(f"{chord_type_names[i]}: {count} ({percentage:.2f}%)")
    
    # Root note statistics
    root_note_counts = np.bincount(chord_roots, minlength=12)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    print("\nRoot Note Distribution:")
    for i, count in enumerate(root_note_counts):
        if i < len(note_names):
            percentage = (count / total_chords) * 100
            print(f"{note_names[i]}: {count} ({percentage:.2f}%)")
    
    # Plot chord type distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(chord_type_counts)), chord_type_counts)
    plt.xticks(range(len(chord_type_names)), chord_type_names, rotation=45)
    plt.title('Chord Type Distribution')
    plt.tight_layout()
    plt.savefig('data/processed/chord_type_distribution.png')
    
    # Plot root note distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(root_note_counts)), root_note_counts)
    plt.xticks(range(len(note_names)), note_names, rotation=45)
    plt.title('Root Note Distribution')
    plt.tight_layout()
    plt.savefig('data/processed/root_note_distribution.png')
    
    # Check melody note distribution
    melody_notes = []
    for seq in melody_sequences:
        melody_notes.extend(seq)
    
    print("\nMelody Note Statistics:")
    print(f"Note range: {min(melody_notes)} - {max(melody_notes)}")
    print(f"Average note value: {np.mean(melody_notes):.2f}")
    
    # Plot melody note distribution
    plt.figure(figsize=(12, 6))
    plt.hist(melody_notes, bins=50)
    plt.title('Melody Note Distribution')
    plt.xlabel('Note Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('data/processed/melody_note_distribution.png')
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print(f"Total number of chords: {total_chords}")
    print(f"Number of unique chord types: {len(set(chord_types))}")
    print(f"Number of unique root notes: {len(set(chord_roots))}")
    print(f"Most common chord type: {chord_type_names[np.argmax(chord_type_counts)]}")
    print(f"Most common root note: {note_names[np.argmax(root_note_counts)]}")
    
    # Print invalid chord statistics
    invalid_roots = [root for root in chord_roots if root >= 12]
    if invalid_roots:
        print(f"\nWarning: Found {len(invalid_roots)} chords with invalid root notes")
        print(f"Invalid root note range: {min(invalid_roots)} - {max(invalid_roots)}")
    
    print("\nData check completed! Charts saved to data/processed directory.")

if __name__ == "__main__":
    check_processed_data() 