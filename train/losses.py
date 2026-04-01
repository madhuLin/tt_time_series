import torch
import torch.nn as nn
import pandas as pd

def compute_class_weights(df, col):
    """
    依資料分佈計算 CrossEntropy 權重
    """
    counts = df[col].value_counts().sort_index()
    # 簡單的逆頻率權重
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights.values)

class MultiTaskLoss(nn.Module):
    def __init__(self, config, action_weights=None, point_weights=None):
        super().__init__()
        self.action_loss = nn.CrossEntropyLoss(weight=action_weights)
        self.point_loss = nn.CrossEntropyLoss(weight=point_weights)
        self.outcome_loss = nn.BCEWithLogitsLoss()
        self.weights = config.LOSS_WEIGHTS

    def forward(self, preds, targets):
        p_act, p_poi, p_out = preds
        t_act, t_poi, t_out = targets
        
        l_act = self.action_loss(p_act, t_act)
        l_poi = self.point_loss(p_poi, t_poi)
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
