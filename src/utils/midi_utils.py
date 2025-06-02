import torch
import pretty_midi
import numpy as np
from pathlib import Path
import music21
from datetime import datetime

def notes_to_midi(notes, output_path, tempo=120, note_duration=0.25):
    """
    Convert note sequence to MIDI file
    Args:
        notes: list or tensor of MIDI note numbers
        output_path: path to save MIDI file
        tempo: tempo in BPM
        note_duration: duration of each note in seconds
    """
    # Convert tensor to list if needed
    if isinstance(notes, torch.Tensor):
        notes = notes.cpu().numpy().tolist()
    
    # Create a PrettyMIDI object
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Create an instrument (piano)
    instrument = pretty_midi.Instrument(program=0)  # Piano
    
    # Add notes
    start_time = 0.0
    for note_number in notes:
        if note_number > 0:  # Skip rest notes (0)
            # Ensure note is in valid MIDI range
            note_number = max(0, min(127, int(note_number)))
            
            note = pretty_midi.Note(
                velocity=64,  # Medium velocity
                pitch=note_number,
                start=start_time,
                end=start_time + note_duration
            )
            instrument.notes.append(note)
        start_time += note_duration
    
    pm.instruments.append(instrument)
    # Convert Path object to string for pretty_midi compatibility
    pm.write(str(output_path))

def create_melody_pair_midi(original_melody, generated_melody, chord_name, output_dir, epoch, pair_idx):
    """
    Create paired MIDI files for original and generated melodies
    Args:
        original_melody: original melody sequence
        generated_melody: generated melody sequence
        chord_name: name of the chord used for generation (can be a progression)
        output_dir: directory to save files
        epoch: current training epoch
        pair_idx: index of the melody pair
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%H%M%S")
    
    # Shorten chord name if too long for filename
    chord_short = chord_name[:20] if len(chord_name) > 20 else chord_name
    chord_short = chord_short.replace('/', '_').replace('#', 'sharp').replace('♭', 'flat')
    
    # Save original melody
    original_path = output_dir / f"e{epoch:03d}_p{pair_idx:02d}_orig_{chord_short}_{timestamp}.mid"
    notes_to_midi(original_melody, original_path)
    
    # Save generated melody
    generated_path = output_dir / f"e{epoch:03d}_p{pair_idx:02d}_gen_{chord_short}_{timestamp}.mid"
    notes_to_midi(generated_melody, generated_path)
    
    return original_path, generated_path

def create_combined_midi(original_melody, generated_melody, output_path, tempo=120):
    """
    Create a single MIDI file with both original and generated melodies on different tracks
    Args:
        original_melody: original melody sequence
        generated_melody: generated melody sequence
        output_path: path to save combined MIDI file
        tempo: tempo in BPM
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Create instruments for original and generated melodies
    original_instrument = pretty_midi.Instrument(program=0, name="Original")  # Piano
    generated_instrument = pretty_midi.Instrument(program=1, name="Generated")  # Bright Piano
    
    note_duration = 0.25
    
    # Add original melody
    start_time = 0.0
    for note_number in original_melody:
        if note_number > 0:
            note_number = max(0, min(127, int(note_number)))
            note = pretty_midi.Note(
                velocity=80,
                pitch=note_number,
                start=start_time,
                end=start_time + note_duration
            )
            original_instrument.notes.append(note)
        start_time += note_duration
    
    # Add generated melody
    start_time = 0.0
    for note_number in generated_melody:
        if note_number > 0:
            note_number = max(0, min(127, int(note_number)))
            note = pretty_midi.Note(
                velocity=60,
                pitch=note_number,
                start=start_time,
                end=start_time + note_duration
            )
            generated_instrument.notes.append(note)
        start_time += note_duration
    
    pm.instruments.append(original_instrument)
    pm.instruments.append(generated_instrument)
    # Convert Path object to string for pretty_midi compatibility
    pm.write(str(output_path))

def batch_generate_samples(model, train_loader, metadata, device, num_samples=5):
    """
    Generate sample melodies from the training data for comparison
    Args:
        model: trained model
        train_loader: training data loader
        metadata: metadata with chord information
        device: computation device
        num_samples: number of samples to generate
    Returns:
        list of (chord_sequence, chord_names, original_melody, generated_melody) tuples
    """
    model.eval()
    samples = []
    
    # Get chord mapping
    id_to_chord = metadata['id_to_chord']
    
    with torch.no_grad():
        for i, (chord_sequences, melody_sequences) in enumerate(train_loader):
            if i >= num_samples:
                break
                
            # Take first sample from batch
            chord_sequence = chord_sequences[0]  # [512] - chord sequence
            original_melody = melody_sequences[0]  # [512] - melody sequence
            
            # Get representative chord names (first few unique chords)
            unique_chords = torch.unique(chord_sequence)[:5]  # Take first 5 unique chords
            chord_names = [id_to_chord[cid.item()] for cid in unique_chords]
            chord_name_str = "_".join(chord_names[:3])  # Use first 3 for filename
            
            # Generate melody using the model
            chord_sequence_device = chord_sequence.to(device)
            generated_melody = model.generate_melody(chord_sequence_device, temperature=0.8)
            
            samples.append((chord_sequence, chord_name_str, original_melody, generated_melody))
    
    return samples

def save_training_samples(model, train_loader, metadata, device, output_dir, epoch, num_samples=5):
    """
    Save sample generation results during training
    Args:
        model: current model
        train_loader: training data loader
        metadata: metadata with chord information
        device: computation device
        output_dir: output directory
        epoch: current epoch
        num_samples: number of samples to generate
    """
    print(f"\n🎵 Generating {num_samples} melody samples at epoch {epoch}...")
    
    samples = batch_generate_samples(model, train_loader, metadata, device, num_samples)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    saved_files = []
    
    for i, (chord_sequence, chord_name_str, original_melody, generated_melody) in enumerate(samples):
        # Create individual MIDI files with chord sequence info
        original_path, generated_path = create_melody_pair_midi(
            original_melody, generated_melody, chord_name_str, output_dir, epoch, i
        )
        
        # Create combined MIDI file
        combined_path = output_dir / f"e{epoch:03d}_p{i:02d}_combo_{chord_name_str[:15]}.mid"
        create_combined_midi(original_melody, generated_melody, combined_path)
        
        saved_files.extend([original_path, generated_path, combined_path])
        
        print(f"  Sample {i+1}: {chord_name_str} -> {len(saved_files)//3} files saved")
    
    print(f"✅ Saved {len(saved_files)} MIDI files to {output_dir}")
    return saved_files 