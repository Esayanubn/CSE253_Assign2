import torch
import torch.nn as nn
import math
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ChordToMelodyTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, chord_vocab_size=CHORD_VOCAB_SIZE, 
                 d_model=D_MODEL, nhead=NHEAD, 
                 num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, 
                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, batch_first=True):
        super().__init__()
        
        # Store dimensions
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.chord_vocab_size = chord_vocab_size
        
        # Separate embeddings for melody notes and chords
        self.melody_embedding = nn.Embedding(vocab_size, d_model)
        self.chord_embedding = nn.Embedding(chord_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=SEQUENCE_LENGTH + 100)
        
        # Encoder processes chord information
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder generates melody
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, chord_sequences, melody_sequence=None):
        """
        Args:
            chord_sequences: [batch_size, seq_len] - chord sequence (time-aligned with melody)
            melody_sequence: [batch_size, seq_len] - melody sequence (for training)
        Returns:
            output: [batch_size, seq_len, vocab_size] - predicted melody logits
        """
        batch_size, seq_len = chord_sequences.size()
        
        # Embed chord sequences
        chord_embedded = self.chord_embedding(chord_sequences)  # [batch_size, seq_len, d_model]
        chord_embedded = self.pos_encoder(chord_embedded)
        
        # Process chord information through encoder
        memory = self.transformer_encoder(chord_embedded)  # [batch_size, seq_len, d_model]
        
        if melody_sequence is not None:
            # Training mode
            # Embed melody sequence
            melody_embedded = self.melody_embedding(melody_sequence)  # [batch_size, seq_len, d_model]
            melody_embedded = self.pos_encoder(melody_embedded)
            
            # Create target mask for decoder (causal mask)
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(melody_sequence.device)
            
            # Decode melody using chord conditioning
            output = self.transformer_decoder(
                melody_embedded,
                memory,
                tgt_mask=tgt_mask
            )
        else:
            # Generation mode - use memory directly (will be processed by output layer)
            output = memory
        
        # Project to vocabulary
        output = self.output_layer(output)  # [batch_size, seq_len, vocab_size]
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_melody(self, chord_sequence, temperature=1.0):
        """
        Generate melody given a chord sequence
        Args:
            chord_sequence: tensor [seq_len] - chord sequence
            temperature: float - sampling temperature
        Returns:
            melody: list - generated melody sequence
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Ensure chord_sequence is 2D
        if chord_sequence.dim() == 1:
            chord_sequence = chord_sequence.unsqueeze(0)  # [1, seq_len]
        
        chord_sequence = chord_sequence.to(device)
        seq_len = chord_sequence.size(1)
        
        # Initialize melody with start token (0)
        melody = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for t in range(1, seq_len):
                # Get current partial melody
                current_melody = melody[:, :t]  # [1, t]
                current_chords = chord_sequence[:, :t]  # [1, t]
                
                # Get predictions for current position
                output = self.forward(current_chords, current_melody)  # [1, t, vocab_size]
                
                # Get last prediction and apply temperature
                logits = output[0, -1, :] / temperature  # [vocab_size]
                probs = torch.softmax(logits, dim=-1)
                
                # Sample next note
                next_note = torch.multinomial(probs, 1).item()
                melody[0, t] = next_note
        
        return melody[0].cpu().tolist()  # Return as list 