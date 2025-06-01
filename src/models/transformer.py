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
    def __init__(self, vocab_size=128, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, batch_first=True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, chord_sequence, melody_sequence=None):
        # chord_sequence: [batch_size, seq_len]
        # melody_sequence: [batch_size, seq_len] (optional)
        
        # Embed and add positional encoding
        chord_embedded = self.embedding(chord_sequence)  # [batch_size, seq_len, d_model]
        chord_embedded = self.pos_encoder(chord_embedded)
        
        # Create encoder output
        memory = self.transformer_encoder(chord_embedded)
        
        if melody_sequence is not None:
            # Training mode
            melody_embedded = self.embedding(melody_sequence)
            melody_embedded = self.pos_encoder(melody_embedded)
            
            # Create target mask for decoder
            tgt_mask = self.generate_square_subsequent_mask(melody_sequence.size(1)).to(melody_sequence.device)
            
            # Decode
            output = self.transformer_decoder(
                melody_embedded,
                memory,
                tgt_mask=tgt_mask
            )
        else:
            # Generation mode
            output = memory
        
        # Project to vocabulary
        output = self.output_layer(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask 