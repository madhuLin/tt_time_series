import torch
from tqdm import tqdm
import numpy as np
from .metrics import calculate_metrics

class Trainer:
    def __init__(self, model, optimizer, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = config.DEVICE

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(loader, desc="Training", leave=False):
            self.optimizer.zero_grad()
            
            cat = batch['cat_features'].to(self.device)
            num = batch['num_features'].to(self.device)
            mask = batch['padding_mask'].to(self.device)
            t_act = batch['target_action'].to(self.device)
            t_poi = batch['target_point'].to(self.device)
            t_out = batch['target_outcome'].to(self.device)
            
            preds = self.model(cat, num, mask)
            loss, _ = self.criterion(preds, (t_act, t_poi, t_out))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        results = {
            'y_true': {'action': [], 'point': [], 'outcome': []},
            'y_pred': {'action': [], 'point': []},
            'y_score': {'outcome': []}
        }
        
        for batch in loader:
            cat = batch['cat_features'].to(self.device)
            num = batch['num_features'].to(self.device)
            mask = batch['padding_mask'].to(self.device)
            
            p_act, p_poi, p_out = self.model(cat, num, mask)
            
            results['y_pred']['action'].extend(p_act.argmax(dim=-1).cpu().numpy())
            results['y_true']['action'].extend(batch['target_action'].numpy())
            
            results['y_pred']['point'].extend(p_poi.argmax(dim=-1).cpu().numpy())
            results['y_true']['point'].extend(batch['target_point'].numpy())
            
            results['y_score']['outcome'].extend(torch.sigmoid(p_out).cpu().numpy())
            results['y_true']['outcome'].extend(batch['target_outcome'].numpy())
            
        return calculate_metrics(results['y_true'], results['y_pred'], results['y_score'])
