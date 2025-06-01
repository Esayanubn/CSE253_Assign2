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
            
            # Convert to binary (note on/off)
            piano_track.binarize()
            
            # Get chord track (index 2 in LPD-5)
            chord_track = pianoroll.tracks[2]
            
            # Process each bar
            for i in range(0, len(piano_track), 16):  # 16 time steps per bar
                if i + 32 > len(piano_track):  # Only process complete sequences
                    break
                    
                # Extract melody (piano track)
                melody = piano_track[i:i+32]
                if not melody.any():  # Skip empty sequences
                    continue
                
                # Extract chord
                chord = chord_track[i:i+32]
                if not chord.any():  # Skip sequences without chords
                    continue
                
                # Convert to sequences
                melody_seq = melody.nonzero()[1]  # Get note indices
                chord_seq = chord.nonzero()[1]    # Get chord indices
                
                # Ensure sequences are within vocabulary size
                if melody_seq.max() >= 128 or chord_seq.max() >= 128:
                    continue
                
                # Convert to lists and ensure consistent length
                melody_seq = melody_seq.tolist()
                chord_seq = chord_seq.tolist()
                
                if len(melody_seq) == 32 and len(chord_seq) == 32:
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