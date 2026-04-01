import torch
import torch.nn as nn
from .embeddings import MultiFeatureEmbedding
from .encoder import PositionalEncoding, TransformerEncoder
from .heads import MultiTaskHeads

class TTMultiTaskModel(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        self.pooling_type = getattr(config, 'POOLING_TYPE', 'last')
        
        self.embedding = MultiFeatureEmbedding(
            vocab_sizes, config.CAT_FEATURES, 
            len(config.NUM_FEATURES), embed_dim=32, d_model=config.D_MODEL
        )
        self.pos_encoder = PositionalEncoding(config.D_MODEL, config.MAX_SEQ_LEN)
        self.encoder = TransformerEncoder(
            config.D_MODEL, config.NHEAD, config.NUM_LAYERS, 
            config.DIM_FEEDFORWARD, config.DROPOUT
        )
        
        # Concat 模式下，輸入維度會翻倍
        head_in_dim = config.D_MODEL * 2 if self.pooling_type == 'concat' else config.D_MODEL
        self.heads = MultiTaskHeads(
            head_in_dim, config.NUM_ACTION_CLASSES, config.NUM_POINT_CLASSES
        )

    def forward(self, cat_feats, num_feats, padding_mask):
        # x: (B, S, D)
        x = self.embedding(cat_feats, num_feats)
        x = self.pos_encoder(x)
        output = self.encoder(x, padding_mask)
        
        if self.pooling_type == 'last':
            # 取最後一個 token 的 hidden state (常用於分類)
            pooled = output[:, -1, :]
            
        elif self.pooling_type == 'mean':
            # Masked Mean Pooling: 只對非 padding 部分取平均
            # padding_mask: (B, S), True=padded
            mask = (~padding_mask).float().unsqueeze(-1) # (B, S, 1)
            sum_hidden = (output * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / count
            
        elif self.pooling_type == 'concat':
            # Last Token + Masked Mean
            last = output[:, -1, :]
            mask = (~padding_mask).float().unsqueeze(-1)
            mean = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.cat([last, mean], dim=-1) # (B, D*2)
            
        else:
            pooled = output[:, -1, :]
            
        return self.heads(pooled)
