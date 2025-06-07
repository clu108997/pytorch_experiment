import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    A Decoder-Only Transformer model similar to GPT.
    It performs masked self-attention to predict the next token in a sequence.
    """
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'GPTLikeTransformer'
        self.d_model = d_model

        self.input_emb = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Define a single Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=False # Set to True if your input batch dimension is first
        )
        
        # Stack multiple decoder layers to form the full decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=nlayers
        )

        self.output_linear = nn.Linear(d_model, ntoken) # Projects decoder output to vocabulary size

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.output_linear.bias)
        nn.init.uniform_(self.output_linear.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with 0s on diag."""
        # This mask ensures that attention is only paid to earlier positions in the sequence.
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (seq_len, batch_size) - input sequence.
        """
        # Generate the causal mask for self-attention in the decoder
        # This is critical for decoder-only models
        device = src.device
        src_mask = self._generate_square_subsequent_mask(len(src)).to(device)

        # 1. Embed the input tokens
        src = self.input_emb(src) * math.sqrt(self.d_model) # Scale embeddings
        
        # 2. Add positional encoding
        src = self.pos_encoder(src)

        # 3. Pass through the Transformer Decoder.
        # The 'tgt' argument is the input sequence itself for masked self-attention.
        # tgt_mask applies the causal mask to the self-attention.
        # memory (encoder_output) is not used in a decoder-only setup, so it's omitted or can be None.
        output = self.transformer_decoder(tgt=src, memory=None, tgt_mask=src_mask)

        # 4. Project the decoder's output to the vocabulary size
        output = self.output_linear(output)
        
        # 5. Apply log_softmax for log probabilities (useful for NLLLoss)
        return F.log_softmax(output, dim=-1)

# Example Usage:
if __name__ == "__main__":
    ntoken = 10000  # Size of vocabulary
    d_model = 512   # Embedding dimension and model dimension
    nhead = 8       # Number of attention heads
    d_hid = 2048    # Dimension of the feedforward network model in nn.TransformerDecoderLayer
    nlayers = 6     # Number of nn.TransformerDecoderLayer in nn.TransformerDecoder
    dropout = 0.2

    model = TransformerModel(ntoken, d_model, nhead, d_hid, nlayers, dropout)

    # Example input: (seq_len, batch_size)
    seq_len = 10
    batch_size = 2
    # Simulate some input token IDs
    input_sequence = torch.randint(0, ntoken, (seq_len, batch_size)) 

    print(f"Input sequence shape: {input_sequence.shape}")

    output = model(input_sequence)
    print(f"Output shape (seq_len, batch_size, ntoken): {output.shape}")

    # Verify that the output is log probabilities
    print(f"Example output probabilities (should sum to approx 1 for each token): {torch.exp(output[0, 0, :10])}")
    print(f"Sum of log probabilities for first token in first sequence: {torch.exp(output[0, 0]).sum()}")