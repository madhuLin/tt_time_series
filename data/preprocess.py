import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple

class Preprocessor:
    def __init__(self, cat_features: List[str]):
        self.encoders = {feat: LabelEncoder() for feat in cat_features}
        self.cat_features = cat_features

    def fit(self, df: pd.DataFrame):
        for feat in self.cat_features:
            # 處理缺失值或測試集新類別，加入 "unknown"
            full_values = pd.concat([df[feat].astype(str), pd.Series(['unknown'])])
            self.encoders[feat].fit(full_values)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        for feat in self.cat_features:
            # 將未知類別 mapping 到 unknown
            res[feat] = res[feat].astype(str).apply(
                lambda x: x if x in self.encoders[feat].classes_ else 'unknown'
            )
            res[feat] = self.encoders[feat].transform(res[feat])
        return res

    def get_vocab_sizes(self) -> Dict[str, int]:
        return {feat: len(enc.classes_) for feat, enc in self.encoders.items()}

def create_sequences(df: pd.DataFrame, max_len: int) -> List[Dict]:
    """
    根據 rally_uid 分組並生成多個 prefix 樣本
    """
    samples = []
    df = df.sort_values(['rally_uid', 'strikeNumber'])
    
    grouped = df.groupby('rally_uid')
    for _, group in grouped:
        n = len(group)
        if n < 2: continue
        
        # serverGetPoint 是整個 rally 的固定結果
        outcome = group['serverGetPoint'].iloc[0]
        
        # 對於長度為 L 的 rally，生成 L-1 個訓練樣本
        for i in range(1, n):
            prefix = group.iloc[:i]
            target_row = group.iloc[i]
            
            samples.append({
                'seq': prefix.tail(max_len).copy(),
                'target_action': target_row['actionId'],
                'target_point': target_row['pointId'],
                'target_outcome': outcome
            })
    return samples
