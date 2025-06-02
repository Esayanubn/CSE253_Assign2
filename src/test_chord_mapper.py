#!/usr/bin/env python3

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

from data.chord_mapper import ChordMapper

def test_chord_mapper():
    """Test the ChordMapper functionality"""
    print("🧪 Testing ChordMapper...")
    
    # Initialize mapper
    mapper = ChordMapper()
    print(f"Initialized mapper with {mapper.num_chords} chord types")
    
    # Test cases
    test_cases = [
        # C major chord: C-E-G (notes 60, 64, 67)
        {
            'name': 'C Major Chord',
            'notes': [60, 64, 67],
            'expected': 'C'
        },
        # A minor chord: A-C-E (notes 57, 60, 64)
        {
            'name': 'A Minor Chord', 
            'notes': [57, 60, 64],
            'expected': 'Am'
        },
        # G7 chord: G-B-D-F (notes 55, 59, 62, 65)
        {
            'name': 'G7 Chord',
            'notes': [55, 59, 62, 65],
            'expected': 'G7'
        },
        # Empty (silence)
        {
            'name': 'Silence',
            'notes': [],
            'expected': 'N'
        },
        # Complex chord with many notes
        {
            'name': 'Complex Chord',
            'notes': [48, 52, 55, 60, 64, 67, 72],  # Should match C major
            'expected': 'C'
        }
    ]
    
    print("\n🎵 Testing individual chords:")
    for i, test_case in enumerate(test_cases, 1):
        # Create 128-dim multi-hot vector
        multihot = np.zeros(128, dtype=np.float32)
        for note in test_case['notes']:
            if 0 <= note < 128:
                multihot[note] = 1.0
        
        # Map to chord
        chord_name, chord_id, confidence = mapper.map_multihot_to_chord(multihot)
        
        # Check result
        result = "✅" if chord_name == test_case['expected'] else "❌"
        print(f"  Test {i}: {test_case['name']}")
        print(f"    Notes: {test_case['notes']}")
        print(f"    Expected: {test_case['expected']}, Got: {chord_name} (ID: {chord_id}, Confidence: {confidence:.3f}) {result}")
    
    # Test batch processing
    print("\n🎼 Testing batch processing:")
    batch_size = 5
    batch_multihot = np.random.randint(0, 2, (batch_size, 128)).astype(np.float32)
    
    chord_names, chord_ids, confidences = mapper.map_batch_multihot_to_chords(batch_multihot)
    
    print(f"Batch size: {batch_size}")
    for i in range(batch_size):
        active_notes = np.where(batch_multihot[i] > 0)[0]
        print(f"  Sample {i+1}: {len(active_notes)} active notes -> {chord_names[i]} (confidence: {confidences[i]:.3f})")
    
    # Test chord progression
    progression = mapper.get_chord_progression_string(chord_names)
    print(f"\nProgression: {progression}")
    
    print("\n✅ ChordMapper test completed!")

def test_with_real_data():
    """Test with real processed data if available"""
    try:
        import torch
        from pathlib import Path
        
        processed_dir = Path("data/processed")
        if not (processed_dir / "metadata.pt").exists():
            print("⚠️ No processed data found. Run prepare_data.py first.")
            return
        
        print("\n🔍 Testing with real processed data:")
        metadata = torch.load(processed_dir / "metadata.pt")
        
        print(f"Total chord types found: {metadata['num_chords']}")
        print(f"Average confidence: {metadata['stats']['avg_chord_confidence']:.3f}")
        
        if 'chord_names_sample' in metadata:
            print("\nSample chord progression from real data:")
            sample_chords = metadata['chord_names_sample'][:10]
            sample_confidences = metadata['confidences_sample'][:10]
            
            for i, (chord, conf) in enumerate(zip(sample_chords, sample_confidences)):
                print(f"  Bar {i+1}: {chord} (confidence: {conf:.3f})")
        
        print("✅ Real data test completed!")
        
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")

if __name__ == "__main__":
    test_chord_mapper()
    test_with_real_data() 