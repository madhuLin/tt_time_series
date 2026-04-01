import torch.nn as nn

class MultiTaskHeads(nn.Module):
    def __init__(self, d_model, num_action_classes, num_point_classes, outcome_in_dim=None):
        super().__init__()
        if outcome_in_dim is None:
            outcome_in_dim = d_model
            
        # Action Head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_action_classes)
        )
        
        # Point Head
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),

            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(d_model // 2, num_point_classes)
        )
        
        # Outcome Head
        self.outcome_head = nn.Sequential(
            nn.Linear(outcome_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x_last, x_outcome):
        logits_action = self.action_head(x_last)
        logits_point = self.point_head(x_last)
        logits_outcome = self.outcome_head(x_outcome).squeeze(-1)
        return logits_action, logits_point, logits_outcome
