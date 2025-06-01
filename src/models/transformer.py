import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ChordToMelodyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1, batch_first=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=batch_first
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, chord_sequence, melody_sequence=None, max_len=None):
        # Encode chord sequence
        chord_embedded = self.embedding(chord_sequence)
        chord_embedded = self.pos_encoder(chord_embedded)
        chord_embedded = self.dropout(chord_embedded)
        
        # Through encoder
        memory = self.transformer_encoder(chord_embedded)
        
        if melody_sequence is not None:
            # Training mode
            melody_embedded = self.embedding(melody_sequence)
            melody_embedded = self.pos_encoder(melody_embedded)
            melody_embedded = self.dropout(melody_embedded)
            
            # Create mask
            tgt_mask = self.generate_square_subsequent_mask(melody_sequence.size(1)).to(melody_sequence.device)
            
            # Through decoder
            output = self.transformer_decoder(
                melody_embedded,
                memory,
                tgt_mask=tgt_mask
            )
        else:
            # Generation mode
            if max_len is None:
                max_len = chord_sequence.size(1)
            
            # Initialize generation sequence
            generated = torch.zeros((chord_sequence.size(0), 1), dtype=torch.long, device=chord_sequence.device)
            
            for _ in range(max_len - 1):
                # Embed current sequence
                current_embedded = self.embedding(generated)
                current_embedded = self.pos_encoder(current_embedded)
                current_embedded = self.dropout(current_embedded)
                
                # Create mask
                tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(generated.device)
                
                # Through decoder
                output = self.transformer_decoder(
                    current_embedded,
                    memory,
                    tgt_mask=tgt_mask
                )
                
                # Predict next note
                next_note = self.decoder(output[:, -1:])
                next_note = torch.argmax(next_note, dim=-1)
                
                # Add to generation sequence
                generated = torch.cat([generated, next_note], dim=1)
            
            return self.decoder(output)
        
        return self.decoder(output) 