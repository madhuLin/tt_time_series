import torch.nn as nn

class MultiTaskHeads(nn.Module):
    def __init__(self, action_in_dim, point_in_dim, outcome_in_dim,
                 hidden_dim, num_action_classes, num_point_classes):
        super().__init__()
        if outcome_in_dim is None:
            outcome_in_dim = d_model
            
        # Action Head
        self.action_head = nn.Sequential(
            nn.Linear(action_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_action_classes)
        )
        
        # Point Head
        self.point_head = nn.Sequential(
            nn.Linear(point_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim // 2, num_point_classes)
        )
        
        # Outcome Head
        self.outcome_head = nn.Sequential(
            nn.Linear(outcome_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_action, x_point, x_outcome):
        logits_action = self.action_head(x_action)
        logits_point = self.point_head(x_point)
        logits_outcome = self.outcome_head(x_outcome).squeeze(-1)
        return logits_action, logits_point, logits_outcome
