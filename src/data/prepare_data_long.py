import os
import numpy as np
import torch
from pathlib import Path
import pypianoroll
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import SEQUENCE_LENGTH, BAR_LENGTH, STEP_SIZE
from chord_mapper import ChordMapper


def process_lpd_files_long():
    """Process LPD files to extract long chord and melody sequences (512 steps)"""
    lpd_dir = Path("data/raw/lpd_5/lpd_5_full/0")
    if not lpd_dir.exists():
        print("Error: LPD dataset not found. Please download it first.")
        return
    
    # Initialize chord mapper
    chord_mapper = ChordMapper()
    print(f"Initialized ChordMapper with {chord_mapper.num_chords} chord types")
    print(f"Using sequence length: {SEQUENCE_LENGTH} steps")
    print(f"Bar length: {BAR_LENGTH} steps")
    print(f"Step size: {STEP_SIZE} steps")
    
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
        'chord_confidence_low': 0,
        'successful': 0
    }
    
    full_list = list(lpd_dir.glob("**/*.npz"))
    subset_list = full_list[:50]  # Process 500 files
    
    print(f"Starting to process {len(subset_list)} files...")
    
    for npz_file in tqdm(subset_list, desc="Processing files"):
        total_files += 1
        file_sequences_added = 0
        file_total_sequences = 0
        
        try:
            multitrack = pypianoroll.load(npz_file)
            tracks = multitrack.tracks  # [0] Drum, [1] Piano, [2] Guitar, [3] Bass, [4] Strings
            
            # Check track count
            if len(tracks) < 5:
                skip_reasons['insufficient_tracks'] += 1
                skipped_files += 1
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
                continue
            
            # Check if length is sufficient for long sequences
            common_length = track_lengths[0]
            if common_length < SEQUENCE_LENGTH:
                skip_reasons['too_short'] += 1
                skipped_files += 1
                continue
            
            # Binarize each track (except drum) - 修复二值化
            for idx in [1, 2, 3, 4]:
                # 手动二值化：将所有非零值设为1
                tracks[idx].pianoroll = (tracks[idx].pianoroll > 0).astype(np.uint8)
            
            # Create long sequences with sliding window
            for i in range(0, common_length - SEQUENCE_LENGTH + 1, STEP_SIZE):
                file_total_sequences += 1
                
                # Extract long sequence (512 steps)
                sequence_notes = np.zeros((SEQUENCE_LENGTH, 128), dtype=np.uint8)

                # Merge all 4 tracks (except Drum)
                for idx in [1, 2, 3, 4]:
                    track_slice = tracks[idx].pianoroll[i:i+SEQUENCE_LENGTH]
                    sequence_notes |= track_slice

                if not sequence_notes.any():
                    sequence_skip_reasons['no_notes'] += 1
                    continue

                # Extract melody from Piano track for the entire sequence
                melody_seq = []
                piano_sequence = tracks[1].pianoroll[i:i+SEQUENCE_LENGTH]  # Piano track
                
                for t in range(SEQUENCE_LENGTH):
                    notes = piano_sequence[t].nonzero()[0]
                    if len(notes) > 0:
                        melody_seq.append(notes[-1].item())  # Use highest note
                    else:
                        melody_seq.append(0)

                # Check for issues
                max_melody_note = max(melody_seq)
                max_sequence_note = np.max(sequence_notes)
                
                if max_melody_note >= 128:
                    sequence_skip_reasons['notes_out_of_range'] += 1
                    continue
                    
                if max_sequence_note > 1:
                    sequence_skip_reasons['binarization_error'] += 1
                    continue
                
                # Apply chord mapping: average over multiple bars to get a representative chord
                # Divide the sequence into bars and get the most representative chord
                bar_chord_votes = []
                
                for bar_start in range(0, SEQUENCE_LENGTH, BAR_LENGTH):
                    bar_end = min(bar_start + BAR_LENGTH, SEQUENCE_LENGTH)
                    bar_notes = sequence_notes[bar_start:bar_end]
                    
                    if bar_notes.any():
                        # Average the multi-hot vector over time to get chord for this bar
                        avg_multihot = np.mean(bar_notes, axis=0)  # Shape: [128]
                        chord_name, chord_id, confidence = chord_mapper.map_multihot_to_chord(avg_multihot)
                        
                        if confidence > 0.3:  # Only count confident chord detections
                            bar_chord_votes.append((chord_id, confidence))
                
                # Choose the most confident chord from all bars
                if not bar_chord_votes:
                    sequence_skip_reasons['chord_confidence_low'] += 1
                    continue
                
                # Get the chord with highest confidence
                best_chord_id, best_confidence = max(bar_chord_votes, key=lambda x: x[1])
                chord_name = chord_mapper.id_to_chord[best_chord_id]
                
                # Store the data
                melody_sequences.append(melody_seq)
                chord_sequences.append(best_chord_id)
                chord_names_list.append(chord_name)
                chord_confidences.append(best_confidence)
                processed_files += 1
                file_sequences_added += 1
                sequence_skip_reasons['successful'] += 1

            # print(f"    📊 {npz_file.name}: {file_sequences_added}/{file_total_sequences} sequences added")

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
    total_sequences_processed = sum(sequence_skip_reasons.values())
    for reason, count in sequence_skip_reasons.items():
        print(f"  {reason}: {count} sequences ({count/total_sequences_processed*100:.1f}%)")

    if not melody_sequences:
        print("⚠️ Warning: No valid sequences found!")
        return

    # Convert to tensors
    melody_tensor = torch.tensor(melody_sequences, dtype=torch.long)  # [N, 512]
    chord_tensor = torch.tensor(chord_sequences, dtype=torch.long)    # [N] - single chord per sequence
    
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
    
    # Save training data with new naming to distinguish from short sequences
    torch.save({
        'chord_sequences': train_chord,
        'melody_sequences': train_melody,
        'split': 'train',
        'sequence_length': SEQUENCE_LENGTH
    }, processed_dir / "train_data_long.pt")
    
    # Save test data
    torch.save({
        'chord_sequences': test_chord,
        'melody_sequences': test_melody,
        'split': 'test',
        'sequence_length': SEQUENCE_LENGTH
    }, processed_dir / "test_data_long.pt")
    
    # Save metadata
    metadata = {
        'chord_to_id': chord_mapper.chord_to_id,
        'id_to_chord': chord_mapper.id_to_chord,
        'chord_templates': chord_mapper.chord_templates,
        'num_chords': chord_mapper.num_chords,
        'sequence_length': SEQUENCE_LENGTH,
        'bar_length': BAR_LENGTH,
        'step_size': STEP_SIZE,
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
        'chord_names_sample': chord_names_list[:100],
        'confidences_sample': [float(x) for x in chord_confidences[:100]]
    }
    
    torch.save(metadata, processed_dir / "metadata_long.pt")

    print(f"\n✅ Long sequence data saved:")
    print(f"  📁 Training data: {processed_dir / 'train_data_long.pt'}")
    print(f"  📁 Test data: {processed_dir / 'test_data_long.pt'}")
    print(f"  📁 Metadata: {processed_dir / 'metadata_long.pt'}")
    print(f"\n📊 Data Shapes:")
    print(f"  🎼 Train melody sequences: {train_melody.shape}")
    print(f"  🎵 Train chord sequences: {train_chord.shape}")
    print(f"  🎼 Test melody sequences: {test_melody.shape}")
    print(f"  🎵 Test chord sequences: {test_chord.shape}")
    print(f"\n⏱️  Sequence duration: {SEQUENCE_LENGTH * 0.125:.1f} seconds per sequence")
    print(f"🎵 Musical bars: {SEQUENCE_LENGTH // BAR_LENGTH} bars per sequence")

if __name__ == "__main__":
    process_lpd_files_long() 