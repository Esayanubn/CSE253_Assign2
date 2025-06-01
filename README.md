# Chord to Melody Generation

This project implements a Transformer-based model for generating melodies from chord progressions. The model is trained on the Lakh Pianoroll Dataset (LPD) and can generate melodies that follow the harmonic structure of the input chords.

## Dataset

We use the Lakh Pianoroll Dataset (LPD-5), which contains 174,154 multitrack pianorolls derived from the Lakh MIDI Dataset. The tracks are merged into five common categories: Drums, Piano, Guitar, Bass, and Strings.

Dataset download link: [Lakh Pianoroll Dataset](https://hermandong.com/lakh-pianoroll-dataset/dataset.html)

## Project Structure

```
Assign2/
├── data/
│   ├── raw/          # Raw LPD dataset
│   └── processed/    # Processed data for training
├── models/           # Saved model checkpoints
├── notebooks/        # Jupyter notebooks for analysis
├── src/
│   ├── data/         # Data processing scripts
│   ├── models/       # Model architecture
│   └── utils/        # Utility functions
└── utils/            # Additional utilities
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and prepare the dataset:
```bash
python src/data/prepare_data.py
```

## Training

To train the model:
```bash
python src/train.py
```

Training parameters can be adjusted in `src/train.py`:
- Batch size: 16
- Sequence length: 32
- Number of epochs: 100
- Learning rate: 0.0005

## Model Architecture

The model uses a Transformer architecture with:
- Encoder-decoder structure
- Multi-head attention
- Positional encoding
- Dropout for regularization

## Data Processing

The data processing pipeline:
1. Downloads and extracts the LPD-5 dataset
2. Processes MIDI files to extract chord and melody sequences
3. Converts musical elements to numerical representations
4. Creates training and validation sets

## Generation

To generate melodies from chords:
```bash
python src/generate.py
```

## Results

The model generates melodies that:
- Follow the harmonic structure of input chords
- Maintain musical coherence
- Exhibit natural melodic progression

## Requirements

- Python 3.8+
- PyTorch 1.7.0+
- NumPy 1.19.0+
- pypianoroll 0.5.0+
- Other dependencies listed in requirements.txt

## License

This project is for educational purposes only.

## References

- [Lakh Pianoroll Dataset](https://hermandong.com/lakh-pianoroll-dataset/dataset.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Music Transformer](https://arxiv.org/abs/1809.04281) 