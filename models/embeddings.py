import torch
import torch.nn as nn

class MultiFeatureEmbedding(nn.Module):
    def __init__(self, vocab_sizes, cat_cols, num_dim, embed_dim, d_model):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_sizes[feat], embed_dim)
            for feat in cat_cols
        })
        
        total_cat_dim = len(cat_cols) * embed_dim
        self.num_projection = nn.Linear(num_dim, embed_dim)
        self.final_projection = nn.Linear(total_cat_dim + embed_dim, d_model)

    def forward(self, cat_feats, num_feats):
        cat_embeds = []
        for i, feat in enumerate(self.embeddings.keys()):
            cat_embeds.append(self.embeddings[feat](cat_feats[:, :, i]))
            
        cat_stack = torch.cat(cat_embeds, dim=-1) # (B, S, total_cat_dim)
        num_proj = self.num_projection(num_feats) # (B, S, embed_dim)
        
        combined = torch.cat([cat_stack, num_proj], dim=-1)
        return self.final_projection(combined)
