import os
import numpy as np
import torch
from pathlib import Path
import pypianoroll
from tqdm import tqdm
import requests
import tarfile
import json
from chord_mapper import ChordMapper


def process_lpd_files():
    """Process LPD files to extract chord and melody sequences with chord mapping"""
    lpd_dir = Path("data/raw/lpd_5/lpd_5_full/0")
    if not lpd_dir.exists():
        print("Error: LPD dataset not found. Please download it first.")
        return
    
    # Initialize chord mapper
    chord_mapper = ChordMapper()
    print(f"Initialized ChordMapper with {chord_mapper.num_chords} chord types")
    
    # Statistics
    melody_sequences = []
    chord_sequences = []  # Now storing chord IDs instead of raw notes
    chord_names_list = []  # Store chord names for analysis
    chord_confidences = []  # Store confidence scores
    processed_files = 0
    skipped_files = 0
    total_files = 0
    
    # Detailed statistics for debugging
    skip_reasons = {
        'insufficient_tracks': 0,
        'inconsistent_lengths': 0,
        'too_short': 0,
        'processing_error': 0
    }
    
    sequence_skip_reasons = {
        'no_notes': 0,
        'notes_out_of_range': 0,
        'binarization_error': 0,
        'successful': 0
    }
    
    full_list = list(lpd_dir.glob("**/*.npz"))
    subset_list = full_list[:10]  # 增加到500个文件
    
    print(f"Starting to process {len(subset_list)} files...")
    
    for npz_file in tqdm(subset_list, desc="Processing files"):
        total_files += 1
        file_sequences_added = 0
        file_total_bars = 0
        
        try:
            multitrack = pypianoroll.load(npz_file)
            tracks = multitrack.tracks  # [0] Drum, [1] Piano, [2] Guitar, [3] Bass, [4] Strings
            
            # Check track count
            if len(tracks) < 5:
                skip_reasons['insufficient_tracks'] += 1
                skipped_files += 1
                # print(f"  ❌ {npz_file.name}: insufficient tracks ({len(tracks)})")
                continue
            
            # Check if all five tracks have the same length
            track_lengths = []
            for idx in range(5):  # Check all 5 tracks
                track_length = tracks[idx].pianoroll.shape[0]
                track_lengths.append(track_length)
            
            # Check if all lengths are the same
            if len(set(track_lengths)) != 1:
                skip_reasons['inconsistent_lengths'] += 1
                skipped_files += 1
                # print(f"  ❌ {npz_file.name}: inconsistent track lengths {track_lengths}")
                continue
            
            # Check if length is sufficient
            common_length = track_lengths[0]
            if common_length < 32:
                skip_reasons['too_short'] += 1
                skipped_files += 1
                # print(f"  ❌ {npz_file.name}: too short ({common_length})")
                continue
            
            # print(f"  ✅ Processing {npz_file.name}: length {common_length}, potential bars: {(common_length - 32) // 16 + 1}")
            
            # Binarize each track (except drum) - 修复二值化
            for idx in [1, 2, 3, 4]:
                # 手动二值化：将所有非零值设为1
                tracks[idx].pianoroll = (tracks[idx].pianoroll > 0).astype(np.uint8)
            
            # Now we can safely process the data since all tracks have the same length
            for i in range(0, common_length, 16):  # Bar-wise split with step=16
                file_total_bars += 1
                
                if i + 32 > common_length:
                    # print(f"    ⚠️  Bar {file_total_bars}: not enough frames ({i}+32 > {common_length})")
                    break
                
                # Note matrix shape: [32, 128] - multi-hot encoding
                bar_notes = np.zeros((32, 128), dtype=np.uint8)

                # Since all tracks have the same length, no need for additional checks
                for idx in [1, 2, 3, 4]:  # Merge all 4 tracks (except Drum)
                    track_slice = tracks[idx].pianoroll[i:i+32]
                    bar_notes |= track_slice

                if not bar_notes.any():
                    sequence_skip_reasons['no_notes'] += 1
                    # print(f"    ❌ Bar {file_total_bars}: no notes detected")
                    continue

                # Extract melody from Piano track
                melody_seq = []
                piano_bar = tracks[1].pianoroll[i:i+32]  # Piano track
                
                for t in range(32):
                    notes = piano_bar[t].nonzero()[0]
                    if len(notes) > 0:
                        melody_seq.append(notes[-1].item())  # Use highest note
                    else:
                        melody_seq.append(0)

                # Check for issues
                max_melody_note = max(melody_seq)
                max_bar_note = np.max(bar_notes)
                
                if max_melody_note >= 128:
                    sequence_skip_reasons['notes_out_of_range'] += 1
                    # print(f"    ❌ Bar {file_total_bars}: melody note out of range (max: {max_melody_note})")
                    continue
                    
                if max_bar_note > 1:
                    sequence_skip_reasons['binarization_error'] += 1
                    # print(f"    ❌ Bar {file_total_bars}: binarization error (max value: {max_bar_note})")
                    continue
                
                # Apply chord mapping: convert 128D multi-hot to chord ID
                # Average the multi-hot vector over time to get chord for the entire bar
                avg_multihot = np.mean(bar_notes, axis=0)  # Shape: [128]
                chord_name, chord_id, confidence = chord_mapper.map_multihot_to_chord(avg_multihot)
                
                # Store the data
                melody_sequences.append(melody_seq)
                chord_sequences.append(chord_id)  # Single chord ID per bar
                chord_names_list.append(chord_name)
                chord_confidences.append(confidence)
                processed_files += 1
                file_sequences_added += 1
                sequence_skip_reasons['successful'] += 1
                
                # Print first few successful bars for debugging
                # if len(melody_sequences) <= 10:
                    # print(f"    ✅ Bar {file_total_bars}: chord={chord_name} (conf={confidence:.3f}), melody_range=[{min(melody_seq)}-{max(melody_seq)}]")

            # print(f"    📊 {npz_file.name}: {file_sequences_added}/{file_total_bars} bars added")

        except Exception as e:
            skip_reasons['processing_error'] += 1
            print(f"  ❌ Error processing {npz_file}: {e}")
            skipped_files += 1
            continue

    # Print detailed statistics
    print(f"\n📊 Detailed Processing Statistics:")
    print(f"Total files: {total_files}")
    print(f"Successfully processed files: {total_files - skipped_files}")
    print(f"Extracted sequences: {len(melody_sequences)}")
    print(f"Average sequences per successful file: {len(melody_sequences) / max(1, total_files - skipped_files):.2f}")
    print(f"Average chord confidence: {np.mean(chord_confidences):.3f}")
    
    print(f"\n❌ File Skip Reasons:")
    for reason, count in skip_reasons.items():
        print(f"  {reason}: {count} files ({count/total_files*100:.1f}%)")
    
    print(f"\n🎼 Sequence Processing Results:")
    total_bars_processed = sum(sequence_skip_reasons.values())
    for reason, count in sequence_skip_reasons.items():
        print(f"  {reason}: {count} bars ({count/total_bars_processed*100:.1f}%)")

    if not melody_sequences:
        print("⚠️ Warning: No valid sequences found!")
        return

    # Convert to tensors
    melody_tensor = torch.tensor(melody_sequences, dtype=torch.long)  # [N, 32]
    chord_tensor = torch.tensor(chord_sequences, dtype=torch.long)    # [N] - single chord per bar
    
    # Print chord distribution
    print(f"\n🎵 Chord Distribution:")
    unique_chords, counts = np.unique(chord_sequences, return_counts=True)
    for chord_id, count in zip(unique_chords, counts):
        chord_name = chord_mapper.id_to_chord[chord_id]
        print(f"  {chord_name}: {count} ({count/len(chord_sequences)*100:.1f}%)")

    # Create train/test split (80/20)
    num_samples = len(melody_sequences)
    num_train = int(0.8 * num_samples)
    
    # Random shuffle
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    # Split data
    train_melody = melody_tensor[train_indices]
    train_chord = chord_tensor[train_indices]
    test_melody = melody_tensor[test_indices]
    test_chord = chord_tensor[test_indices]
    
    # Save processed data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Save training data
    torch.save({
        'chord_sequences': train_chord,
        'melody_sequences': train_melody,
        'split': 'train'
    }, processed_dir / "train_data.pt")
    
    # Save test data
    torch.save({
        'chord_sequences': test_chord,
        'melody_sequences': test_melody,
        'split': 'test'
    }, processed_dir / "test_data.pt")
    
    # Save metadata without custom objects - only basic data types
    metadata = {
        'chord_to_id': chord_mapper.chord_to_id,  # Dict[str, int]
        'id_to_chord': chord_mapper.id_to_chord,  # Dict[int, str]
        'chord_templates': chord_mapper.chord_templates,  # Dict[str, List[int]]
        'num_chords': chord_mapper.num_chords,  # int
        'stats': {
            'total_files': total_files,
            'skipped_files': skipped_files,
            'processed_sequences': len(melody_sequences),
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'avg_chord_confidence': float(np.mean(chord_confidences)),
            'skip_reasons': skip_reasons,
            'sequence_skip_reasons': sequence_skip_reasons
        },
        'chord_names_sample': chord_names_list[:100],  # List[str]
        'confidences_sample': [float(x) for x in chord_confidences[:100]]  # List[float]
    }
    
    torch.save(metadata, processed_dir / "metadata.pt")

    print(f"\n✅ Data saved:")
    print(f"  📁 Training data: {processed_dir / 'train_data.pt'}")
    print(f"  📁 Test data: {processed_dir / 'test_data.pt'}")
    print(f"  📁 Metadata: {processed_dir / 'metadata.pt'}")
    print(f"\n📊 Data Shapes:")
    print(f"  🎼 Train melody sequences: {train_melody.shape}")
    print(f"  🎵 Train chord sequences: {train_chord.shape}")
    print(f"  🎼 Test melody sequences: {test_melody.shape}")
    print(f"  🎵 Test chord sequences: {test_chord.shape}")

# Additional helper function to check data integrity
def check_track_integrity(npz_file):
    """Check track data integrity"""
    try:
        multitrack = pypianoroll.load(npz_file)
        tracks = multitrack.tracks
        
        print(f"\nFile: {npz_file.name}")
        print(f"Track count: {len(tracks)}")
        
        for i, track in enumerate(tracks):
            print(f"Track {i} ({track.name}): shape {track.pianoroll.shape}")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def load_processed_data():
    """Load processed training and test data"""
    processed_dir = Path("data/processed")
    
    train_data = torch.load(processed_dir / "train_data.pt", weights_only=True)
    test_data = torch.load(processed_dir / "test_data.pt", weights_only=True)
    metadata = torch.load(processed_dir / "metadata.pt", weights_only=True)
    
    return train_data, test_data, metadata

def analyze_chord_data():
    """Analyze the processed chord data"""
    try:
        _, _, metadata = load_processed_data()
        
        print("🎵 Chord Analysis:")
        print(f"Total chord types: {metadata['num_chords']}")
        print(f"Average confidence: {metadata['stats']['avg_chord_confidence']:.3f}")
        
        print("\nSample chord progression:")
        chord_names = metadata['chord_names_sample'][:10]
        confidences = metadata['confidences_sample'][:10]
        for i, (chord, conf) in enumerate(zip(chord_names, confidences)):
            print(f"  Bar {i+1}: {chord} (confidence: {conf:.3f})")
            
    except FileNotFoundError:
        print("❌ Processed data not found. Please run process_lpd_files() first.")

def create_chord_mapper_from_metadata(metadata):
    """Recreate ChordMapper from saved metadata"""
    chord_mapper = ChordMapper()
    # Verify the saved data matches our chord mapper
    assert chord_mapper.chord_to_id == metadata['chord_to_id']
    assert chord_mapper.id_to_chord == metadata['id_to_chord']
    assert chord_mapper.num_chords == metadata['num_chords']
    return chord_mapper

if __name__ == "__main__":
    # download_lpd_dataset()
    process_lpd_files()
    analyze_chord_data() 