import os
import numpy as np
import torch
from pathlib import Path
import pypianoroll
from tqdm import tqdm
import requests
import tarfile
import json


def process_lpd_files():
    """Process LPD files to extract chord and melody sequences"""
    lpd_dir = Path("data/raw/lpd_5/lpd_5_full/0")
    if not lpd_dir.exists():
        print("Error: LPD dataset not found. Please download it first.")
        return
    
    chord_sequences = []
    melody_sequences = []
    processed_files = 0
    
    # Process each .npz file
    for npz_file in tqdm(list(lpd_dir.glob("**/*.npz"))):
        try:
            # Load the pianoroll
            pianoroll = pypianoroll.load(npz_file)
            
            # Get piano track (index 1 in LPD-5)
            piano_track = pianoroll.tracks[1]
            piano_track.binarize()
            
            # Get chord track (index 2 in LPD-5)
            chord_track = pianoroll.tracks[2]
            
            # Process each bar
            for i in range(0, len(piano_track), 16):  # 16 time steps per bar
                if i + 32 > len(piano_track):  # Only process complete sequences
                    break
                    
                # Extract melody and chord for this bar
                melody = piano_track[i:i+32]
                chord = chord_track[i:i+32]
                
                if not melody.any() or not chord.any():  # Skip empty sequences
                    continue
                
                # Process each time step
                melody_seq = []
                chord_seq = []
                
                for t in range(32):
                    # Get melody note at current time step
                    melody_notes = melody[t].nonzero()[0]
                    if len(melody_notes) > 0:
                        melody_seq.append(melody_notes[0].item())  # Take highest note
                    else:
                        melody_seq.append(0)  # Rest
                    
                    # Get chord at current time step
                    chord_notes = chord[t].nonzero()[0]
                    if len(chord_notes) > 0:
                        chord_seq.append(chord_notes[0].item())  # Take root note
                    else:
                        chord_seq.append(0)  # No chord
                
                # Ensure sequences are within vocabulary size
                if max(melody_seq) >= 128 or max(chord_seq) >= 128:
                    continue
                
                chord_sequences.append(chord_seq)
                melody_sequences.append(melody_seq)
            
            processed_files += 1
            
        except Exception as e:
            print(f"Error processing {npz_file}: {str(e)}")
            continue
    
    print(f"Processed {processed_files} files")
    print(f"Total sequences: {len(chord_sequences)}")
    
    if len(chord_sequences) == 0:
        print("Warning: No valid sequences found!")
        return
    
    # Convert to tensors
    chord_sequences = torch.tensor(chord_sequences, dtype=torch.long)
    melody_sequences = torch.tensor(melody_sequences, dtype=torch.long)
    
    # Save processed data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    torch.save({
        'chord_sequences': chord_sequences,
        'melody_sequences': melody_sequences
    }, processed_dir / "processed_data.pt")
    
    print(f"Average sequence length: {len(chord_sequences[0])}")

def chord_to_sequence(chord):
    """将和弦转换为数值序列"""
    # 获取和弦的根音
    root = min(chord)
    # 计算和弦类型
    intervals = [pitch - root for pitch in chord]
    chord_type = get_chord_type(intervals)
    return root * 10 + chord_type

def get_chord_type(intervals):
    """根据音程判断和弦类型"""
    intervals = sorted(intervals)
    if len(intervals) == 3:
        if intervals == [0, 4, 7]:
            return 1  # 大三和弦
        elif intervals == [0, 3, 7]:
            return 2  # 小三和弦
    elif len(intervals) == 4:
        if intervals == [0, 4, 7, 11]:
            return 3  # 大七和弦
        elif intervals == [0, 3, 7, 10]:
            return 4  # 小七和弦
        elif intervals == [0, 4, 7, 10]:
            return 5  # 属七和弦
    return 0  # 其他类型

if __name__ == "__main__":
    # download_lpd_dataset()
    process_lpd_files() 