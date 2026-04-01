import torch
from torch.utils.data import Dataset
import numpy as np

class TableTennisDataset(Dataset):
    def __init__(self, samples, cat_cols, num_cols, max_len, vocab_sizes):
        self.samples = samples
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_len = max_len
        self.vocab_sizes = vocab_sizes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq_df = s['seq']
        
        cat_data = seq_df[self.cat_cols].values
        num_data = seq_df[self.num_cols].values
        
        curr_len = len(cat_data)
        pad_len = self.max_len - curr_len
        
        # 前補零 (Pre-padding)
        if pad_len > 0:
            # 每個類別特徵使用其 vocab_size 作為 padding index
            cat_pad = np.array([self.vocab_sizes[feat] for feat in self.cat_cols])
            cat_pad = np.tile(cat_pad, (pad_len, 1))
            
            num_pad = np.zeros((pad_len, len(self.num_cols)), dtype=float)
            cat_data = np.vstack([cat_pad, cat_data])
            num_data = np.vstack([num_pad, num_data])
            mask = [True] * pad_len + [False] * curr_len
        else:
            # 即使不補零也確保 copy 以避免 non-writable 警告
            cat_data = cat_data.copy()
            num_data = num_data.copy()
            mask = [False] * self.max_len
            
        return {
            'cat_features': torch.LongTensor(cat_data),
            'num_features': torch.FloatTensor(num_data),
            'padding_mask': torch.BoolTensor(mask),
            'target_action': torch.tensor(s['target_action'], dtype=torch.long),
            'target_point': torch.tensor(s['target_point'], dtype=torch.long),
            'target_outcome': torch.tensor(s['target_outcome'], dtype=torch.float)
        }
