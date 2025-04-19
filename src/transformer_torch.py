import torch
import torch.nn as nn
import random

class RandomTransformer(nn.Module):
    def __init__(self, vocab_size=1000, max_seq_len=128):
        super().__init__()

        # Random hyperparameters
        self.embed_dim = random.choice([64, 128, 256, 512])
        self.num_heads = random.choice([2, 4, 8])
        self.ff_dim = self.embed_dim * random.choice([2, 4])
        self.num_layers = random.randint(1, 6)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, self.embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            activation=random.choice(['relu', 'gelu'])
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output layer
        self.output_layer = nn.Linear(self.embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.encoder(x)
        return self.output_layer(x)

# --- Example usage ---
if __name__ == "__main__":
    model = RandomTransformer(vocab_size=5000, max_seq_len=128)
    print("Random Transformer Architecture:")
    print(model)

    # Dummy input: batch_size=4, sequence_length=128
    dummy_input = torch.randint(0, 5000, (4, 128))  # (batch, seq_len)
    output = model(dummy_input)

    print("\nOutput shape:", output.shape)  # (batch, seq_len, vocab_size)

