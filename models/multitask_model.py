import torch
import torch.nn as nn
from .embeddings import MultiFeatureEmbedding
from .encoder import PositionalEncoding, TransformerEncoder
from .heads import MultiTaskHeads

class TTMultiTaskModel(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        self.pooling_type = getattr(config, 'POOLING_TYPE', 'concat')
        
        self.embedding = MultiFeatureEmbedding(
            vocab_sizes, config.CAT_FEATURES, 
            len(config.NUM_FEATURES), embed_dim=32, d_model=config.D_MODEL
        )
        self.pos_encoder = PositionalEncoding(config.D_MODEL, config.MAX_SEQ_LEN)
        self.encoder = TransformerEncoder(
            config.D_MODEL, config.NHEAD, config.NUM_LAYERS, 
            config.DIM_FEEDFORWARD, config.DROPOUT
        )
        
        # 不同的 Head 輸入維度可能不同
        self.heads = MultiTaskHeads(
            config.D_MODEL, # 用於 Action/Point (Last Token)
            config.NUM_ACTION_CLASSES, 
            config.NUM_POINT_CLASSES,
            outcome_in_dim=config.D_MODEL * 2 if self.pooling_type == 'concat' else config.D_MODEL
        )

    def forward(self, cat_feats, num_feats, padding_mask):
        x = self.embedding(cat_feats, num_feats)
        x = self.pos_encoder(x)
        output = self.encoder(x, padding_mask)
        
        # 1. 取最後一拍用於 Action/Point 預測
        last_hidden = output[:, -1, :]
        
        # 2. 計算 Outcome 特徵
        if self.pooling_type == 'concat':
            mask = (~padding_mask).float().unsqueeze(-1)
            mean_hidden = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outcome_hidden = torch.cat([last_hidden, mean_hidden], dim=-1)
        elif self.pooling_type == 'mean':
            mask = (~padding_mask).float().unsqueeze(-1)
            outcome_hidden = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            outcome_hidden = last_hidden
            
        return self.heads(last_hidden, outcome_hidden)
