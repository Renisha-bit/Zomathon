import torch
import torch.nn as nn

class CSAORailModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_dim=10):
        super().__init__()
        
        # 1. Embeddings for Items and Context
        self.item_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.context_embeddings = nn.Linear(context_dim, embed_dim)

        # 2. Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.cart_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 3. Final Ranking Head
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 3, 128), 
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, cart_items, user_context, candidate_item):
        # Process Cart through Transformer
        cart_repr = self.item_embeddings(cart_items)
        cart_repr = self.cart_transformer(cart_repr).mean(dim=1)

        # Process Candidate Item
        cand_repr = self.item_embeddings(candidate_item)
        
        # Process Context Features
        ctx_repr = torch.relu(self.context_embeddings(user_context))

        # Combine all 3 representations
        combined = torch.cat([cart_repr, cand_repr, ctx_repr], dim=-1)
        return self.fc(combined)


# ==========================================
# FOR TESTING
# ==========================================
if __name__ == "__main__":
    V_SIZE = 1000
    E_DIM = 32
    
    # Initialize model
    model = CSAORailModel(vocab_size=V_SIZE, embed_dim=E_DIM)
    print("Model initialized successfully!")
    
    # Simulate a batch of 2 requests
    mock_cart = torch.tensor([[101, 102], [105, 0]])    
    mock_context = torch.randn(2, 10)                  
    mock_candidate = torch.tensor([502, 901])          
    
    # Run forward pass
    predictions = model(mock_cart, mock_context, mock_candidate)
    
    print("Forward pass successful!")
    print("Output Scores Shape:", predictions.shape)  # Should print torch.Size([2, 1])
    print("Sample Scores:\n", predictions)
