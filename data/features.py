import pandas as pd
import numpy as np

BOS_POINT = 10      # pointId: 0~9
BOS_ACTION = 19     # actionId: 0~18
BOS_SPIN = 6        # spinId: 0~5
BOS_HAND = 3        # handId: 0~2

def get_action_group(action_id: int) -> int:
    # 根據常見 ID (1-7: Attack, 8-11: Control, 12-14: Defensive, 15-18: Serve)
    if 1 <= action_id <= 7:
        return 1  # Attack
    if 8 <= action_id <= 11:
        return 2  # Control
    if 12 <= action_id <= 14:
        return 3  # Defensive
    if 15 <= action_id <= 18:
        return 4  # Serve
    return 0      # Other / zero

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['rally_uid', 'strikeNumber']).reset_index(drop=True)

    groups = df.groupby('rally_uid', sort=False)

    # 1. Action group
    df['action_group'] = df['actionId'].apply(get_action_group).astype(int)

    # 2. Five-phase feature
    df['phase_id'] = 5
    df.loc[df['strikeNumber'] == 1, 'phase_id'] = 1  # Service
    df.loc[df['strikeNumber'] == 2, 'phase_id'] = 2  # Receive
    df.loc[df['strikeNumber'] == 3, 'phase_id'] = 3  # Third-ball
    df.loc[df['strikeNumber'] == 4, 'phase_id'] = 4  # Fourth-ball
    df['phase_id'] = df['phase_id'].astype(int)

    # 3. Score features
    df['scoreDiff'] = df['scoreSelf'] - df['scoreOther']
    df['absScoreDiff'] = (df['scoreSelf'] - df['scoreOther']).abs()
    df['is_close_score'] = (df['absScoreDiff'] <= 1).astype(int)
    df['is_game_point_like'] = (
        ((df['scoreSelf'] >= 10) | (df['scoreOther'] >= 10)) &
        (df['absScoreDiff'] >= 1)
    ).astype(int)

    # 4. Rally context (無洩漏)
    df['log_strike'] = np.log1p(df['strikeNumber'])
    df['is_late_rally'] = (df['strikeNumber'] > 8).astype(int)

    # 最近 3 拍的攻擊密度 (作為回合結束的訊號)
    df['is_atk_binary'] = (df['action_group'] == 1).astype(int)
    df['recent_atk_density'] = groups['is_atk_binary'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # 5. Parity feature
    df['is_odd_stroke'] = (df['strikeNumber'] % 2 == 1).astype(int)

    # 6. Markov features with BOS token
    df['prev_pointId'] = groups['pointId'].shift(1).fillna(BOS_POINT).astype(int)
    df['prev_actionId'] = groups['actionId'].shift(1).fillna(BOS_ACTION).astype(int)
    df['prev_spinId'] = groups['spinId'].shift(1).fillna(BOS_SPIN).astype(int)
    df['prev_handId'] = groups['handId'].shift(1).fillna(BOS_HAND).astype(int)

    df = df.drop(columns=['is_atk_binary'])
    return df