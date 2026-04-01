import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

def compute_class_weights(df, col, num_classes, max_weight=5.0):
    counts = df[col].value_counts()
    full_counts = counts.reindex(range(num_classes), fill_value=0).astype(float)

    # 沒出現的類別先當成 1，避免權重爆掉
    safe_counts = full_counts.copy()
    safe_counts[safe_counts == 0] = 1.0

    weights = 1.0 / torch.log1p(torch.tensor(safe_counts.values, dtype=torch.float32))
    weights = weights / weights.mean()
    weights = torch.clamp(weights, max=max_weight)

    return weights

class SoftF1Loss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        logits: (B, C)
        targets: (B,)
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        tp = (probs * targets_one_hot).sum(dim=0)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)

        soft_f1 = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        # Macro Soft-F1 Loss
        return 1 - soft_f1.mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, config, action_weights=None, point_weights=None):
        super().__init__()
        self.config = config
        self.action_ce = nn.CrossEntropyLoss(weight=action_weights)
        self.point_ce = nn.CrossEntropyLoss(weight=point_weights)
        self.outcome_loss = nn.BCEWithLogitsLoss()
        
        self.use_soft_f1_action = getattr(config, 'USE_SOFT_F1_ACTION', False)
        self.use_soft_f1_point = getattr(config, 'USE_SOFT_F1_POINT', False)
        self.lambda_f1 = getattr(config, 'LAMBDA_SOFT_F1', 0.1)

        if self.use_soft_f1_action:
            self.action_f1 = SoftF1Loss(num_classes=config.NUM_ACTION_CLASSES)
        if self.use_soft_f1_point:
            self.point_f1 = SoftF1Loss(num_classes=config.NUM_POINT_CLASSES)

        self.weights = config.LOSS_WEIGHTS

    def forward(self, preds, targets):
        p_act, p_poi, p_out = preds
        t_act, t_poi, t_out = targets
        
        # Action Loss
        l_act = self.action_ce(p_act, t_act)
        if self.use_soft_f1_action:
            l_act += self.lambda_f1 * self.action_f1(p_act, t_act)
            
        # Point Loss
        l_poi = self.point_ce(p_poi, t_poi)
        if self.use_soft_f1_point:
            l_poi += self.lambda_f1 * self.point_f1(p_poi, t_poi)
            
        # Outcome Loss
        l_out = self.outcome_loss(p_out, t_out)
        
        total = (self.weights['action'] * l_act + 
                 self.weights['point'] * l_poi + 
                 self.weights['outcome'] * l_out)
        
        return total, {
            'loss': total.item(),
            'action_loss': l_act.item(), 
            'point_loss': l_poi.item(), 
            'outcome_loss': l_out.item()
        }
