import torch
import numpy as np
from pathlib import Path
from models.transformer import ChordToMelodyTransformer
from utils.midi_utils import notes_to_midi, create_combined_midi
from config import *
import random

def load_model_and_metadata(model_path='models/best_model.pth'):
    """Load trained model and metadata"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load metadata (try long sequence metadata first)
    try:
        metadata = torch.load('data/processed/metadata_long.pt', weights_only=True)
        num_chords = metadata['num_chords']
        chord_to_id = metadata['chord_to_id']
        id_to_chord = metadata['id_to_chord']
        print("Loaded long sequence metadata")
    except FileNotFoundError:
        try:
            metadata = torch.load('data/processed/metadata.pt', weights_only=True)
            num_chords = metadata['num_chords']
            chord_to_id = metadata['chord_to_id']
            id_to_chord = metadata['id_to_chord']
            print("Loaded short sequence metadata")
        except FileNotFoundError:
            print("Warning: No metadata found, using default values")
            num_chords = CHORD_VOCAB_SIZE
            chord_to_id = {}
            id_to_chord = {}
    
    # Load model using config values
    model = ChordToMelodyTransformer(
        vocab_size=VOCAB_SIZE,
        chord_vocab_size=num_chords,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    else:
        print(f"Warning: Model file {model_path} not found, using randomly initialized model")
    
    return model, metadata, device

def generate_melody_from_chord_name(model, chord_name, metadata, device, temperature=0.8, repeat_count=32):
    """Generate melody from chord name by creating a repeating chord sequence"""
    chord_to_id = metadata['chord_to_id']
    
    if chord_name in chord_to_id:
        chord_id = chord_to_id[chord_name]
        
        # Create a chord sequence by repeating the chord (simulate a progression)
        chord_sequence = torch.full((SEQUENCE_LENGTH,), chord_id, dtype=torch.long, device=device)
        
        melody = model.generate_melody(chord_sequence, temperature=temperature)
        return melody, chord_id
    else:
        available_chords = list(chord_to_id.keys())[:10]  # Show first 10
        print(f"Chord '{chord_name}' not found. Available chords include: {available_chords}")
        return None, None

def create_chord_progression(chord_names, metadata, device):
    """Create a chord sequence from a progression of chord names"""
    chord_to_id = metadata['chord_to_id']
    
    # Each chord lasts for a certain number of steps
    steps_per_chord = SEQUENCE_LENGTH // len(chord_names)
    chord_sequence = []
    
    for chord_name in chord_names:
        if chord_name in chord_to_id:
            chord_id = chord_to_id[chord_name]
            chord_sequence.extend([chord_id] * steps_per_chord)
        else:
            print(f"Warning: Chord '{chord_name}' not found, using C")
            chord_id = chord_to_id.get('C', 0)
            chord_sequence.extend([chord_id] * steps_per_chord)
    
    # Pad to full length if needed
    while len(chord_sequence) < SEQUENCE_LENGTH:
        chord_sequence.append(chord_sequence[-1])
    
    return torch.tensor(chord_sequence[:SEQUENCE_LENGTH], dtype=torch.long, device=device)

def generate_random_samples(model, metadata, device, num_samples=5, temperature=0.8):
    """Generate random melody samples using different chords"""
    id_to_chord = metadata['id_to_chord']
    samples = []
    
    # Get random chord IDs
    available_chord_ids = list(id_to_chord.keys())
    selected_chord_ids = random.sample(available_chord_ids, min(num_samples, len(available_chord_ids)))
    
    for chord_id in selected_chord_ids:
        chord_name = id_to_chord[chord_id]
        
        # Create chord sequence by repeating the chord
        chord_sequence = torch.full((SEQUENCE_LENGTH,), chord_id, dtype=torch.long, device=device)
        
        melody = model.generate_melody(chord_sequence, temperature=temperature)
        samples.append((chord_id, chord_name, melody))
        print(f"Generated melody for {chord_name} (ID: {chord_id})")
    
    return samples

def save_generated_samples(samples, output_dir='output/generated'):
    """Save generated samples as MIDI files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for i, (chord_id, chord_name, melody) in enumerate(samples):
        # Clean chord name for filename
        clean_chord_name = chord_name.replace('/', '_').replace('#', 'sharp').replace('♭', 'flat')
        
        # Save as MIDI
        midi_path = output_dir / f"generated_{i+1:02d}_{clean_chord_name}.mid"
        notes_to_midi(melody, midi_path)
        saved_files.append(midi_path)
        
        print(f"Saved: {midi_path}")
    
    return saved_files

def interactive_generation(model, metadata, device):
    """Interactive chord-to-melody generation"""
    chord_to_id = metadata['chord_to_id']
    
    print("\n🎵 Interactive Chord-to-Melody Generation")
    print("Available commands:")
    print("  - Enter a chord name (e.g., 'C', 'Am', 'G7', 'Cmaj7')")
    print("  - 'list' to see available chords")
    print("  - 'random' to generate random samples")
    print("  - 'quit' to exit")
    
    output_dir = Path("output/interactive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_count = 0
    
    while True:
        user_input = input("\nEnter chord name or command: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'list':
            print("Available chords:")
            for i, chord in enumerate(sorted(chord_to_id.keys())):
                print(f"  {chord}", end="")
                if (i + 1) % 8 == 0:  # New line every 8 chords
                    print()
            print()
        elif user_input.lower() == 'random':
            samples = generate_random_samples(model, metadata, device, num_samples=3)
            for chord_id, chord_name, melody in samples:
                generated_count += 1
                filename = output_dir / f"random_{generated_count:03d}_{chord_name.replace('/', '_')}.mid"
                notes_to_midi(melody, filename)
                print(f"  Saved: {filename}")
        else:
            melody, chord_id = generate_melody_from_chord_name(model, user_input, metadata, device)
            if melody is not None:
                generated_count += 1
                clean_name = user_input.replace('/', '_').replace('#', 'sharp').replace('♭', 'flat')
                filename = output_dir / f"interactive_{generated_count:03d}_{clean_name}.mid"
                notes_to_midi(melody, filename)
                print(f"  Generated melody for '{user_input}' -> {filename}")
                print(f"  Melody notes: {melody[:8]}... (showing first 8 notes)")

def main():
    print("🎼 Chord-to-Melody Generator")
    print("Loading model and metadata...")
    
    model, metadata, device = load_model_and_metadata()
    
    if metadata:
        print(f"Loaded {metadata['num_chords']} chord types")
        print(f"Dataset stats: {metadata['stats']['processed_sequences']} sequences")
        print(f"Average chord confidence: {metadata['stats']['avg_chord_confidence']:.3f}")
    
    # Generate some examples
    print("\n🎵 Generating example melodies...")
    samples = generate_random_samples(model, metadata, device, num_samples=3)
    save_generated_samples(samples)
    
    # Interactive mode
    try:
        interactive_generation(model, metadata, device)
    except KeyboardInterrupt:
        print("\n\nGeneration stopped by user.")
    
    print("🎵 Generation complete!")

def generate_specific_chords(chord_names, output_dir='output/specific'):
    """Generate melodies for specific chord names"""
    model, metadata, device = load_model_and_metadata()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for chord_name in chord_names:
        melody, chord_id = generate_melody_from_chord_name(model, chord_name, metadata, device)
        if melody is not None:
            clean_name = chord_name.replace('/', '_').replace('#', 'sharp').replace('♭', 'flat')
            filename = output_dir / f"{clean_name}.mid"
            notes_to_midi(melody, filename)
            results.append((chord_name, melody, filename))
            print(f"Generated {chord_name} -> {filename}")
    
    return results

if __name__ == "__main__":
    main() 