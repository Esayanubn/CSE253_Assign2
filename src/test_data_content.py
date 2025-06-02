#!/usr/bin/env python3
"""
Test script to examine the processed data content
"""
import torch
import numpy as np
from pathlib import Path
import sys
# sys.path.append('src')

def load_and_examine_data():
    """Load and examine the processed data"""
    print("🔍 Loading and examining processed data...")
    print("=" * 60)
    
    try:
        # Load training data
        train_data = torch.load('data/processed/train_data.pt', weights_only=True)
        test_data = torch.load('data/processed/test_data.pt', weights_only=True)
        metadata = torch.load('data/processed/metadata.pt', weights_only=True)
        
        print(f"✅ Data loaded successfully!")
        print(f"Training samples: {len(train_data['chord_sequences'])}")
        print(f"Test samples: {len(test_data['chord_sequences'])}")
        print(f"Total chord types: {metadata['num_chords']}")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Extract data
    train_chords = train_data['chord_sequences']
    train_melodies = train_data['melody_sequences']
    chord_to_id = metadata['chord_to_id']
    id_to_chord = metadata['id_to_chord']
    
    print(f"\n📊 Data shapes:")
    print(f"Train chord sequences: {train_chords.shape}")
    print(f"Train melody sequences: {train_melodies.shape}")
    print(f"Chord ID range: [{train_chords.min()}, {train_chords.max()}]")
    print(f"Melody note range: [{train_melodies.min()}, {train_melodies.max()}]")
    
    # Show first few samples
    print(f"\n🎵 First 10 training samples:")
    print("-" * 60)
    
    for i in range(min(10, len(train_chords))):
        chord_id = train_chords[i].item()
        chord_name = id_to_chord[chord_id]
        melody = train_melodies[i].numpy()
        
        print(f"Sample {i+1:2d}: Chord={chord_name:8s} (ID:{chord_id:2d})")
        print(f"           Melody: {melody[:16]}...")  # First 16 notes
        print(f"           Full:   {melody}")
        
        # Calculate some statistics
        non_zero_notes = melody[melody > 0]
        if len(non_zero_notes) > 0:
            print(f"           Stats:  {len(non_zero_notes):2d} notes, range [{non_zero_notes.min():3d}-{non_zero_notes.max():3d}], avg={non_zero_notes.mean():.1f}")
        else:
            print(f"           Stats:  No notes (all zeros)")
        print()
    
    # Show chord distribution
    print(f"\n🎼 Chord distribution (top 15):")
    print("-" * 40)
    
    unique_chords, counts = np.unique(train_chords.numpy(), return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]  # Sort by count, descending
    
    for i, idx in enumerate(sorted_indices[:15]):
        chord_id = unique_chords[idx]
        chord_name = id_to_chord[chord_id]
        count = counts[idx]
        percentage = count / len(train_chords) * 100
        print(f"{i+1:2d}. {chord_name:8s} (ID:{chord_id:2d}): {count:4d} samples ({percentage:5.1f}%)")
    
    # Show melody note distribution
    print(f"\n🎹 Melody note distribution:")
    print("-" * 40)
    
    all_melody_notes = train_melodies.flatten().numpy()
    non_zero_notes = all_melody_notes[all_melody_notes > 0]
    
    print(f"Total melody notes: {len(all_melody_notes)}")
    print(f"Non-zero notes: {len(non_zero_notes)} ({len(non_zero_notes)/len(all_melody_notes)*100:.1f}%)")
    print(f"Zero notes (rests): {len(all_melody_notes) - len(non_zero_notes)} ({(len(all_melody_notes) - len(non_zero_notes))/len(all_melody_notes)*100:.1f}%)")
    
    if len(non_zero_notes) > 0:
        print(f"Note range: [{non_zero_notes.min()}-{non_zero_notes.max()}]")
        print(f"Average note: {non_zero_notes.mean():.1f}")
        print(f"Most common notes:")
        
        unique_notes, note_counts = np.unique(non_zero_notes, return_counts=True)
        sorted_note_indices = np.argsort(note_counts)[::-1]
        
        for i, idx in enumerate(sorted_note_indices[:10]):
            note = unique_notes[idx]
            count = note_counts[idx]
            percentage = count / len(non_zero_notes) * 100
            
            # Convert MIDI note to note name
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            octave = note // 12 - 1
            note_name = note_names[note % 12] + str(octave)
            
            print(f"  {i+1:2d}. Note {note:3d} ({note_name:4s}): {count:5d} times ({percentage:5.1f}%)")

def examine_specific_chord(chord_name):
    """Examine all sequences for a specific chord"""
    try:
        train_data = torch.load('data/processed/train_data.pt', weights_only=True)
        metadata = torch.load('data/processed/metadata.pt', weights_only=True)
        
        chord_to_id = metadata['chord_to_id']
        id_to_chord = metadata['id_to_chord']
        
        if chord_name not in chord_to_id:
            print(f"❌ Chord '{chord_name}' not found!")
            print(f"Available chords: {list(chord_to_id.keys())[:10]}...")
            return
        
        chord_id = chord_to_id[chord_name]
        train_chords = train_data['chord_sequences']
        train_melodies = train_data['melody_sequences']
        
        # Find all sequences with this chord
        chord_mask = train_chords == chord_id
        chord_indices = torch.where(chord_mask)[0]
        
        print(f"\n🎵 All sequences for chord '{chord_name}' (ID: {chord_id}):")
        print(f"Found {len(chord_indices)} sequences")
        print("-" * 60)
        
        for i, idx in enumerate(chord_indices[:5]):  # Show first 5
            melody = train_melodies[idx].numpy()
            non_zero_notes = melody[melody > 0]
            
            print(f"Sequence {i+1:2d} (index {idx:4d}):")
            print(f"  Melody: {melody}")
            if len(non_zero_notes) > 0:
                print(f"  Stats:  {len(non_zero_notes):2d} notes, range [{non_zero_notes.min():3d}-{non_zero_notes.max():3d}]")
            else:
                print(f"  Stats:  No notes")
            print()
        
        if len(chord_indices) > 5:
            print(f"... and {len(chord_indices) - 5} more sequences")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🎼 Data Content Examination Tool")
    print("=" * 50)
    
    # Load and examine all data
    load_and_examine_data()
    
    # Examine specific chords
    print(f"\n" + "=" * 50)
    print("🎵 Examining specific chords:")
    
    test_chords = ['C', 'Am', 'G', 'F', 'Dm']
    for chord in test_chords:
        examine_specific_chord(chord)

if __name__ == "__main__":
    main() 