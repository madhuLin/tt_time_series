import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    實作衍生特徵邏輯
    """
    # 數值特徵計算
    df['scoreDiff'] = df['scoreSelf'] - df['scoreOther']
    df['is_deuce'] = ((df['scoreSelf'] >= 10) & (df['scoreOther'] >= 10)).astype(int)
    df['is_close_score'] = (df['scoreDiff'].abs() <= 2).astype(int)
    df['is_third_stroke'] = (df['strikeNumber'] == 3).astype(int)
    
    # phase_id 分段
    def get_phase(n):
        if n == 1: return 1 # serve
        if n == 2: return 2 # receive
        if n == 3: return 3 # third_ball
        if n == 4: return 4 # fourth_ball
        if 5 <= n <= 8: return 5 # mid_rally
        if n > 8: return 6 # late_rally
        return 0 # padding placeholder
    
    df['phase_id'] = df['strikeNumber'].apply(get_phase)
    return df
