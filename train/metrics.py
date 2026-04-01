from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

def calculate_metrics(y_true, y_pred, y_score):
    """
    計算指定指標並返回加權後的總分
    """
    f1_act = f1_score(y_true['action'], y_pred['action'], average='macro')
    f1_poi = f1_score(y_true['point'], y_pred['point'], average='macro')
    auc_out = roc_auc_score(y_true['outcome'], y_score['outcome'])
    
    overall = 0.4 * f1_act + 0.4 * f1_poi + 0.2 * auc_out
    
    return {
        'action_f1': f1_act,
        'point_f1': f1_poi,
        'outcome_auc': auc_out,
        'overall': overall
    }
