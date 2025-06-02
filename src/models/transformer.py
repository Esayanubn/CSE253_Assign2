import torch
import torch.nn as nn
import math

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
    def __init__(self, vocab_size=128, chord_vocab_size=49, d_model=256, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, 
                 dropout=0.1, batch_first=True):
        super().__init__()
        
        # Separate embeddings for melody notes and chords
        self.melody_embedding = nn.Embedding(vocab_size, d_model)
        self.chord_embedding = nn.Embedding(chord_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
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
        
    def forward(self, chord_ids, melody_sequence=None):
        """
        Args:
            chord_ids: [batch_size] - single chord ID per sequence
            melody_sequence: [batch_size, seq_len] - melody sequence (for training)
        Returns:
            output: [batch_size, seq_len, vocab_size] - predicted melody logits
        """
        batch_size = chord_ids.size(0)
        
        # Embed chord IDs and expand to create conditioning
        chord_embedded = self.chord_embedding(chord_ids)  # [batch_size, d_model]
        
        if melody_sequence is not None:
            # Training mode
            seq_len = melody_sequence.size(1)
            
            # Expand chord embedding to sequence length for encoder input
            chord_sequence = chord_embedded.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, d_model]
            chord_sequence = self.pos_encoder(chord_sequence)
            
            # Process chord information through encoder
            memory = self.transformer_encoder(chord_sequence)  # [batch_size, seq_len, d_model]
            
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
            # Generation mode - create a simple memory from chord
            seq_len = 32  # Default sequence length for generation
            chord_sequence = chord_embedded.unsqueeze(1).repeat(1, seq_len, 1)
            chord_sequence = self.pos_encoder(chord_sequence)
            memory = self.transformer_encoder(chord_sequence)
            output = memory  # Will be processed by output layer
        
        # Project to vocabulary
        output = self.output_layer(output)  # [batch_size, seq_len, vocab_size]
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_melody(self, chord_id, max_length=32, temperature=1.0):
        """
        Generate melody given a chord ID
        Args:
            chord_id: int - single chord ID
            max_length: int - maximum melody length
            temperature: float - sampling temperature
        Returns:
            melody: list - generated melody sequence
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Convert chord_id to tensor
        chord_ids = torch.tensor([chord_id], dtype=torch.long, device=device)
        
        # Initialize melody with start token (0)
        melody = [0]
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Convert current melody to tensor
                melody_tensor = torch.tensor([melody], dtype=torch.long, device=device)
                
                # Get predictions
                output = self.forward(chord_ids, melody_tensor)  # [1, seq_len, vocab_size]
                
                # Get last prediction and apply temperature
                logits = output[0, -1, :] / temperature  # [vocab_size]
                probs = torch.softmax(logits, dim=-1)
                
                # Sample next note
                next_note = torch.multinomial(probs, 1).item()
                melody.append(next_note)
        
        return melody[1:]  # Remove start token 