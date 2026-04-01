import torch
import torch.nn as nn
from .embeddings import MultiFeatureEmbedding
from .encoder import PositionalEncoding, TransformerEncoder
from .heads import MultiTaskHeads

class TTMultiTaskModel(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        self.pooling_type = getattr(config, 'POOLING_TYPE', 'concat')
        d_model = config.D_MODEL
        d_small = d_model // 4  # 這裡定義 small 的維度
        
        self.embedding = MultiFeatureEmbedding(
            vocab_sizes, config.CAT_FEATURES, 
            len(config.NUM_FEATURES), embed_dim=32, d_model=d_model
        )
        self.pos_encoder = PositionalEncoding(d_model, config.MAX_SEQ_LEN)
        self.encoder = TransformerEncoder(
            d_model, config.NHEAD, config.NUM_LAYERS, 
            config.DIM_FEEDFORWARD, config.DROPOUT
        )
        
        # 用於生成 x_outcome_small 的投影層
        self.small_projection = nn.Linear(d_model, d_small)
        
        # 不同的 Head 輸入維度
        # Action/Point: concat(x_last, x_outcome_small) -> d_model + d_small
        # Outcome: concat(x_last, x_mean) -> d_model * 2
        self.heads = MultiTaskHeads(
            d_model + d_small, 
            config.NUM_ACTION_CLASSES, 
            config.NUM_POINT_CLASSES,
            outcome_in_dim=d_model * 2 if self.pooling_type == 'concat' else d_model
        )

    def forward(self, cat_feats, num_feats, padding_mask):
        x = self.embedding(cat_feats, num_feats)
        x = self.pos_encoder(x)
        output = self.encoder(x, padding_mask)
        
        # 1. 取最後一拍 (x_last)
        last_hidden = output[:, -1, :]
        
        # 2. 計算全回合平均 (x_mean)
        mask = (~padding_mask).float().unsqueeze(-1)
        mean_hidden = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # 3. 生成 x_outcome_small (投影後的平均特徵)
        x_outcome_small = self.small_projection(mean_hidden)
        
        # 4. 依照需求拼接
        # Action/Point: concat(x_last, x_outcome_small)
        x_action_point = torch.cat([last_hidden, x_outcome_small], dim=-1)
        
        # Outcome: x_outcome (concat(x_last, x_mean))
        x_outcome = torch.cat([last_hidden, mean_hidden], dim=-1)
            
        return self.heads(x_action_point, x_outcome)
