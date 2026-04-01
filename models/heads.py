import torch.nn as nn

class MultiTaskHeads(nn.Module):
    def __init__(self, d_model, num_action_classes, num_point_classes):
        super().__init__()
        
        # Action Head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_action_classes)
        )
        
        # Point Head
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_point_classes)
        )
        
        # Outcome Head
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x is the hidden state of the last token: (B, d_model)
        logits_action = self.action_head(x)
        logits_point = self.point_head(x)
        logits_outcome = self.outcome_head(x).squeeze(-1)
        return logits_action, logits_point, logits_outcome
