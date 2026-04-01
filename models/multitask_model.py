import torch.nn as nn
from .embeddings import MultiFeatureEmbedding
from .encoder import PositionalEncoding, TransformerEncoder
from .heads import MultiTaskHeads

class TTMultiTaskModel(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        self.embedding = MultiFeatureEmbedding(
            vocab_sizes, config.CAT_FEATURES, 
            len(config.NUM_FEATURES), embed_dim=32, d_model=config.D_MODEL
        )
        self.pos_encoder = PositionalEncoding(config.D_MODEL, config.MAX_SEQ_LEN)
        self.encoder = TransformerEncoder(
            config.D_MODEL, config.NHEAD, config.NUM_LAYERS, 
            config.DIM_FEEDFORWARD, config.DROPOUT
        )
        self.heads = MultiTaskHeads(
            config.D_MODEL, config.NUM_ACTION_CLASSES, config.NUM_POINT_CLASSES
        )

    def forward(self, cat_feats, num_feats, padding_mask):
        x = self.embedding(cat_feats, num_feats)
        x = self.pos_encoder(x)
        output = self.encoder(x, padding_mask)
        
        # Take the last token's representation
        last_hidden = output[:, -1, :]
        return self.heads(last_hidden)
