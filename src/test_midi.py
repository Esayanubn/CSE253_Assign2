#!/usr/bin/env python3
"""
Test script for MIDI generation functionality
"""
import torch
import numpy as np
from pathlib import Path
import sys
# sys.path.append('src')

from utils.midi_utils import notes_to_midi, create_melody_pair_midi

def test_midi_generation():
    """Test basic MIDI file generation"""
    print("🎵 Testing MIDI generation...")
    
    # Create test output directory
    output_dir = Path("output/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test melody - simple C major scale
    test_melody = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    
    # Test basic MIDI generation
    test_file = output_dir / "test_scale.mid"
    try:
        notes_to_midi(test_melody, test_file)
        print(f"✅ Basic MIDI generation successful: {test_file}")
    except Exception as e:
        print(f"❌ Basic MIDI generation failed: {e}")
        return False
    
    # Test paired MIDI generation
    original_melody = [60, 64, 67, 72, 67, 64, 60, 0]  # C major arpeggio
    generated_melody = [62, 65, 69, 74, 69, 65, 62, 0]  # D minor arpeggio
    
    try:
        original_path, generated_path = create_melody_pair_midi(
            original_melody, generated_melody, "Cmaj", output_dir, 1, 0
        )
        print(f"✅ Paired MIDI generation successful:")
        print(f"  Original: {original_path}")
        print(f"  Generated: {generated_path}")
    except Exception as e:
        print(f"❌ Paired MIDI generation failed: {e}")
        return False
    
    return True

def test_metadata_loading():
    """Test metadata loading"""
    print("\n📊 Testing metadata loading...")
    
    try:
        metadata = torch.load('data/processed/metadata.pt', weights_only=True)
        print(f"✅ Metadata loaded successfully")
        print(f"  Number of chords: {metadata['num_chords']}")
        print(f"  Total sequences: {metadata['stats']['processed_sequences']}")
        print(f"  Sample chords: {list(metadata['chord_to_id'].keys())[:5]}")
        return True
    except Exception as e:
        print(f"❌ Metadata loading failed: {e}")
        return False

def main():
    print("🧪 MIDI Generation Test Suite")
    print("=" * 40)
    
    # Test MIDI generation
    midi_ok = test_midi_generation()
    
    # Test metadata loading
    metadata_ok = test_metadata_loading()
    
    print("\n" + "=" * 40)
    if midi_ok and metadata_ok:
        print("✅ All tests passed! Ready for training with MIDI generation.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return midi_ok and metadata_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 