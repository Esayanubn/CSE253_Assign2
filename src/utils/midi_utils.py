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
        chord_name: name of the chord used for generation
        output_dir: directory to save files
        epoch: current training epoch
        pair_idx: index of the melody pair
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%H%M%S")
    
    # Save original melody
    original_path = output_dir / f"epoch_{epoch:03d}_pair_{pair_idx:02d}_original_{chord_name}_{timestamp}.mid"
    notes_to_midi(original_melody, original_path)
    
    # Save generated melody
    generated_path = output_dir / f"epoch_{epoch:03d}_pair_{pair_idx:02d}_generated_{chord_name}_{timestamp}.mid"
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
        list of (chord_id, chord_name, original_melody, generated_melody) tuples
    """
    model.eval()
    samples = []
    
    # Get chord mapping
    id_to_chord = metadata['id_to_chord']
    
    with torch.no_grad():
        for i, (chord_ids, melody_sequences) in enumerate(train_loader):
            if i >= num_samples:
                break
                
            # Take first sample from batch
            chord_id = chord_ids[0].item()
            original_melody = melody_sequences[0]
            chord_name = id_to_chord[chord_id]
            
            # Generate melody using the model
            chord_tensor = torch.tensor([chord_id], device=device)
            generated_melody = model.generate_melody(chord_id, max_length=32, temperature=0.8)
            
            samples.append((chord_id, chord_name, original_melody, generated_melody))
    
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
    
    for i, (chord_id, chord_name, original_melody, generated_melody) in enumerate(samples):
        # Create individual MIDI files
        original_path, generated_path = create_melody_pair_midi(
            original_melody, generated_melody, chord_name, output_dir, epoch, i
        )
        
        # Create combined MIDI file
        combined_path = output_dir / f"epoch_{epoch:03d}_pair_{i:02d}_combined_{chord_name}.mid"
        create_combined_midi(original_melody, generated_melody, combined_path)
        
        saved_files.extend([original_path, generated_path, combined_path])
        
        print(f"  Sample {i+1}: {chord_name} -> {len(saved_files)//3} files saved")
    
    print(f"✅ Saved {len(saved_files)} MIDI files to {output_dir}")
    return saved_files 