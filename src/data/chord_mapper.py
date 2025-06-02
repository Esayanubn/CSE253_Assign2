import numpy as np
from typing import List, Tuple, Dict

class ChordMapper:
    """Map 128-dimensional multi-hot encoding to chord labels"""
    
    def __init__(self):
        # Define common chord patterns (in terms of MIDI note intervals)
        self.chord_templates = {
            # Major chords
            'C': [0, 4, 7],
            'C#': [1, 5, 8], 'Db': [1, 5, 8],
            'D': [2, 6, 9],
            'D#': [3, 7, 10], 'Eb': [3, 7, 10],
            'E': [4, 8, 11],
            'F': [5, 9, 0],
            'F#': [6, 10, 1], 'Gb': [6, 10, 1],
            'G': [7, 11, 2],
            'G#': [8, 0, 3], 'Ab': [8, 0, 3],
            'A': [9, 1, 4],
            'A#': [10, 2, 5], 'Bb': [10, 2, 5],
            'B': [11, 3, 6],
            
            # Minor chords
            'Cm': [0, 3, 7],
            'C#m': [1, 4, 8], 'Dbm': [1, 4, 8],
            'Dm': [2, 5, 9],
            'D#m': [3, 6, 10], 'Ebm': [3, 6, 10],
            'Em': [4, 7, 11],
            'Fm': [5, 8, 0],
            'F#m': [6, 9, 1], 'Gbm': [6, 9, 1],
            'Gm': [7, 10, 2],
            'G#m': [8, 11, 3], 'Abm': [8, 11, 3],
            'Am': [9, 0, 4],
            'A#m': [10, 1, 5], 'Bbm': [10, 1, 5],
            'Bm': [11, 2, 6],
            
            # Dominant 7th chords
            'C7': [0, 4, 7, 10],
            'D7': [2, 6, 9, 0],
            'E7': [4, 8, 11, 2],
            'F7': [5, 9, 0, 3],
            'G7': [7, 11, 2, 5],
            'A7': [9, 1, 4, 7],
            'B7': [11, 3, 6, 9],
            
            # Major 7th chords
            'Cmaj7': [0, 4, 7, 11],
            'Dmaj7': [2, 6, 9, 1],
            'Emaj7': [4, 8, 11, 3],
            'Fmaj7': [5, 9, 0, 4],
            'Gmaj7': [7, 11, 2, 6],
            'Amaj7': [9, 1, 4, 8],
            'Bmaj7': [11, 3, 6, 10],
            
            # Add rest/silence
            'N': []  # No chord (silence)
        }
        
        # Create reverse mapping: chord_name -> chord_id
        self.chord_to_id = {chord: idx for idx, chord in enumerate(self.chord_templates.keys())}
        self.id_to_chord = {idx: chord for chord, idx in self.chord_to_id.items()}
        
        self.num_chords = len(self.chord_templates)
        
    def extract_notes_from_multihot(self, multihot_vector: np.ndarray) -> List[int]:
        """Extract active notes from 128-dimensional multi-hot vector"""
        return np.where(multihot_vector > 0)[0].tolist()
    
    def normalize_chord_to_root_position(self, notes: List[int]) -> List[int]:
        """Normalize chord to root position (within one octave)"""
        if not notes:
            return []
        
        # Convert to pitch classes (0-11)
        pitch_classes = [note % 12 for note in notes]
        # Remove duplicates and sort
        pitch_classes = sorted(list(set(pitch_classes)))
        
        return pitch_classes
    
    def calculate_chord_similarity(self, input_notes: List[int], template_notes: List[int]) -> float:
        """Calculate similarity between input notes and chord template"""
        if not input_notes and not template_notes:
            return 1.0
        if not input_notes or not template_notes:
            return 0.0
        
        input_set = set(input_notes)
        template_set = set(template_notes)
        
        # Calculate Jaccard similarity
        intersection = len(input_set.intersection(template_set))
        union = len(input_set.union(template_set))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def map_multihot_to_chord(self, multihot_vector: np.ndarray) -> Tuple[str, int, float]:
        """
        Map 128-dimensional multi-hot vector to the most matching chord
        
        Returns:
            chord_name: Name of the matched chord
            chord_id: Numerical ID of the chord
            confidence: Similarity score (0-1)
        """
        # Extract active notes
        active_notes = self.extract_notes_from_multihot(multihot_vector)
        
        # If no notes are active, return silence
        if not active_notes:
            return 'N', self.chord_to_id['N'], 1.0
        
        # Normalize to pitch classes
        pitch_classes = self.normalize_chord_to_root_position(active_notes)
        
        best_chord = 'N'
        best_similarity = 0.0
        
        # Find best matching chord
        for chord_name, template_notes in self.chord_templates.items():
            if chord_name == 'N':  # Skip silence when we have notes
                continue
                
            similarity = self.calculate_chord_similarity(pitch_classes, template_notes)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_chord = chord_name
        
        # If similarity is too low, treat as unrecognized chord
        if best_similarity < 0.5:
            best_chord = 'N'
            best_similarity = 0.0
        
        return best_chord, self.chord_to_id[best_chord], best_similarity
    
    def map_batch_multihot_to_chords(self, multihot_batch: np.ndarray) -> Tuple[List[str], List[int], List[float]]:
        """
        Map a batch of multi-hot vectors to chords
        
        Args:
            multihot_batch: Shape [batch_size, time_steps, 128] or [batch_size, 128]
        
        Returns:
            chord_names: List of chord names
            chord_ids: List of chord IDs  
            confidences: List of confidence scores
        """
        if len(multihot_batch.shape) == 3:
            # [batch_size, time_steps, 128] - average over time steps
            multihot_batch = np.mean(multihot_batch, axis=1)
        
        chord_names = []
        chord_ids = []
        confidences = []
        
        for i in range(multihot_batch.shape[0]):
            chord_name, chord_id, confidence = self.map_multihot_to_chord(multihot_batch[i])
            chord_names.append(chord_name)
            chord_ids.append(chord_id)
            confidences.append(confidence)
        
        return chord_names, chord_ids, confidences
    
    def get_chord_progression_string(self, chord_names: List[str]) -> str:
        """Convert list of chord names to a progression string"""
        return ' | '.join(chord_names)
