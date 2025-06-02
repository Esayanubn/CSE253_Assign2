
"""
Global configuration for the chord-to-melody model
"""

# Sequence parameters
SEQUENCE_LENGTH = 512  # Length of melody sequences (previously 32)
BAR_LENGTH = 16       # Length of one musical bar in time steps
STEP_SIZE = 16        # Step size for sliding window when creating sequences

# Model parameters
VOCAB_SIZE = 128      # MIDI note range (0-127)
CHORD_VOCAB_SIZE = 49 # Number of chord types
D_MODEL = 256         # Transformer embedding dimension
NHEAD = 8            # Number of attention heads
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# Training parameters
BATCH_SIZE = 4        # Reduced batch size due to longer sequences
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
GRADIENT_CLIP = 1.0

# MIDI generation parameters
MIDI_TEMPO = 120
NOTE_DURATION = 0.125  # Duration of each time step in seconds (8th notes)

# File paths
DATA_DIR = "data"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "output"
MODELS_DIR = "models"

# Training settings
SAVE_CHECKPOINT_EVERY = 10  # Save checkpoint every N epochs
GENERATE_MIDI_EVERY = 5     # Generate MIDI samples every N epochs
NUM_GENERATION_SAMPLES = 5  # Number of samples to generate

print(f"Configuration loaded:")
print(f"  Sequence length: {SEQUENCE_LENGTH} steps")
print(f"  Bar length: {BAR_LENGTH} steps")
print(f"  Total bars per sequence: {SEQUENCE_LENGTH // BAR_LENGTH}")
print(f"  Sequence duration: {SEQUENCE_LENGTH * NOTE_DURATION:.1f} seconds")
print(f"  Model parameters: {D_MODEL}d, {NHEAD}h, {NUM_ENCODER_LAYERS}+{NUM_DECODER_LAYERS} layers")
print(f"  Batch size: {BATCH_SIZE}") 