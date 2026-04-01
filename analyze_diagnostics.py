import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
from config import Config
from data.features import engineer_features
from data.preprocess import Preprocessor, create_sequences
from data.dataset import TableTennisDataset
from models.multitask_model import TTMultiTaskModel
from torch.utils.data import DataLoader
import os

def run_analysis():
    # 1. 載入資料與特徵工程
    print("--- Data Loading & Engineering ---")
    df = pd.read_csv("train.csv")
    df = engineer_features(df)
    
    # 2. 分佈分析
    print("\n[1] pointId Overall Distribution (Top 10):")
    print(df['pointId'].value_counts(normalize=True).head(10))
    
    print("\n[2] actionId Overall Distribution (Top 10):")
    print(df['actionId'].value_counts(normalize=True).head(10))

    # 3. pointId=0 比例分析
    print("\n[3] pointId=0 Ratio Analysis:")
    p0_by_strike = df.groupby('strikeNumber')['pointId'].apply(lambda x: (x == 0).mean()).head(10)
    print("By strikeNumber (First 10):\n", p0_by_strike)
    
    p0_by_sid = df.groupby('strikeId')['pointId'].apply(lambda x: (x == 0).mean())
    print("By strikeId:\n", p0_by_sid)

    # 4. actionId=0 比例分析
    print("\n[4] actionId=0 Ratio Analysis:")
    a0_by_strike = df.groupby('strikeNumber')['actionId'].apply(lambda x: (x == 0).mean()).head(10)
    print("By strikeNumber (First 10):\n", a0_by_strike)
    
    a0_by_sid = df.groupby('strikeId')['actionId'].apply(lambda x: (x == 0).mean())
    print("By strikeId:\n", a0_by_sid)

    # 5. 模型預測評估 (如果模型存在)
    model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print("\n[!] No best_model.pt found. Skipping F1 analysis.")
        return

    print("\n--- Model Evaluation (Excluding Class 0) ---")
    # 模擬 main.py 的切分與編碼
    matches = df['match'].unique()
    train_matches = matches[:int(len(matches) * 0.8)]
    train_df = df[df['match'].isin(train_matches)].reset_index(drop=True)
    valid_df = df[~df['match'].isin(train_matches)].reset_index(drop=True)
    
    preprocessor = Preprocessor(Config.CAT_FEATURES)
    preprocessor.fit(train_df)
    valid_df_encoded = preprocessor.transform(valid_df)
    
    # 數值特徵縮放 (簡單處理)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    num_cols = Config.NUM_FEATURES
    train_num = engineer_features(train_df)[num_cols]
    scaler.fit(train_num)
    valid_df_encoded[num_cols] = scaler.transform(valid_df_encoded[num_cols])

    valid_samples = create_sequences(valid_df_encoded, Config.MAX_SEQ_LEN)
    vocab_sizes = preprocessor.get_vocab_sizes()
    Config.NUM_ACTION_CLASSES = vocab_sizes['actionId']
    Config.NUM_POINT_CLASSES = vocab_sizes['pointId']
    
    ds = TableTennisDataset(valid_samples, Config.CAT_FEATURES, Config.NUM_FEATURES, Config.MAX_SEQ_LEN, vocab_sizes)
    loader = DataLoader(ds, batch_size=256)
    
    model = TTMultiTaskModel(Config, vocab_sizes).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()
    
    y_true_act, y_pred_act = [], []
    y_true_poi, y_pred_poi = [], []
    
    with torch.no_grad():
        for batch in loader:
            cat = batch['cat_features'].to(Config.DEVICE)
            num = batch['num_features'].to(Config.DEVICE)
            mask = batch['padding_mask'].to(Config.DEVICE)
            
            p_act, p_poi, _ = model(cat, num, mask)
            
            y_true_act.extend(batch['target_action'].numpy())
            y_pred_act.extend(p_act.argmax(dim=-1).cpu().numpy())
            
            y_true_poi.extend(batch['target_point'].numpy())
            y_pred_poi.extend(p_poi.argmax(dim=-1).cpu().numpy())
            
    # 計算排除 0 之後的 F1
    y_true_act, y_pred_act = np.array(y_true_act), np.array(y_pred_act)
    y_true_poi, y_pred_poi = np.array(y_true_poi), np.array(y_pred_poi)
    
    # 取得類別 0 在 LabelEncoder 中的索引 (通常是 0)
    act0_idx = preprocessor.encoders['actionId'].transform(['0'])[0]
    poi0_idx = preprocessor.encoders['pointId'].transform(['0'])[0]
    
    mask_act = (y_true_act != act0_idx)
    mask_poi = (y_true_poi != poi0_idx)
    
    f1_act_no0 = f1_score(y_true_act[mask_act], y_pred_act[mask_act], average='macro')
    f1_poi_no0 = f1_score(y_true_poi[mask_poi], y_pred_poi[mask_poi], average='macro')
    
    print(f"\n[5] Action Macro-F1 (Excluding class 0): {f1_act_no0:.4f}")
    print(f"[6] Point Macro-F1 (Excluding class 0): {f1_poi_no0:.4f}")
    print(f"Compare with All classes (Action): {f1_score(y_true_act, y_pred_act, average='macro'):.4f}")
    print(f"Compare with All classes (Point): {f1_score(y_true_poi, y_pred_poi, average='macro'):.4f}")

if __name__ == "__main__":
    run_analysis()
