import argparse
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader

from config import Config
from data.features import engineer_features
from data.preprocess import Preprocessor, create_sequences
from data.dataset import TableTennisDataset
from models.multitask_model import TTMultiTaskModel
from train.losses import MultiTaskLoss, compute_class_weights
from train.trainer import Trainer
from utils.common import seed_everything

def main(args):
    seed_everything(Config.SEED)
    
    # 建立輸出目錄
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. 資料讀取與特徵工程
    print(f"Loading data from {args.train_csv}...")
    df = pd.read_csv(args.train_csv)
    df = engineer_features(df)
    
    # 2. 資料切割 (Group Split by Match)
    matches = df['match'].unique()
    train_matches = matches[:int(len(matches) * (1 - args.valid_ratio))]
    train_df = df[df['match'].isin(train_matches)].reset_index(drop=True)
    valid_df = df[~df['match'].isin(train_matches)].reset_index(drop=True)
    
    # 3. 預處理與標籤編碼
    preprocessor = Preprocessor(Config.CAT_FEATURES)
    preprocessor.fit(train_df)
    
    train_df_encoded = preprocessor.transform(train_df)
    valid_df_encoded = preprocessor.transform(valid_df)
    
    # 數值特徵縮放
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_df_encoded[Config.NUM_FEATURES] = scaler.fit_transform(train_df_encoded[Config.NUM_FEATURES])
    valid_df_encoded[Config.NUM_FEATURES] = scaler.transform(valid_df_encoded[Config.NUM_FEATURES])
    
    # 4. 生成序列樣本
    print("Creating prefix sequences...")
    train_samples = create_sequences(train_df_encoded, args.max_seq_len)
    valid_samples = create_sequences(valid_df_encoded, args.max_seq_len)
    
    vocab_sizes = preprocessor.get_vocab_sizes()
    
    train_ds = TableTennisDataset(train_samples, Config.CAT_FEATURES, Config.NUM_FEATURES, args.max_seq_len, vocab_sizes)
    valid_ds = TableTennisDataset(valid_samples, Config.CAT_FEATURES, Config.NUM_FEATURES, args.max_seq_len, vocab_sizes)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=2)
    
    # 5. 模型初始化
    # 動態更新類別數
    Config.NUM_ACTION_CLASSES = vocab_sizes['actionId']
    Config.NUM_POINT_CLASSES = vocab_sizes['pointId']
    
    model = TTMultiTaskModel(Config, vocab_sizes).to(Config.DEVICE)
    
    # 6. 損失函數與優化器
    act_w = compute_class_weights(train_df_encoded, 'actionId', vocab_sizes['actionId']).to(Config.DEVICE)
    poi_w = compute_class_weights(train_df_encoded, 'pointId', vocab_sizes['pointId']).to(Config.DEVICE)
    
    criterion = MultiTaskLoss(Config, action_weights=act_w, point_weights=poi_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    
    # 7. 訓練流程
    trainer = Trainer(model, optimizer, criterion, Config)
    
    print(f"Starting training on {Config.DEVICE}...")
    best_score = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        metrics = trainer.evaluate(valid_loader)
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | "
              f"ActF1: {metrics['action_f1']:.4f} | PoiF1: {metrics['point_f1']:.4f} | "
              f"AUC: {metrics['outcome_auc']:.4f} | Overall: {metrics['overall']:.4f}")
        
        # Checkpoint
        if metrics['overall'] > best_score:
            best_score = metrics['overall']
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train.csv", help="訓練資料路徑")
    parser.add_argument("--valid_ratio", type=float, default=0.2, help="驗證集比例")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--max_seq_len", type=int, default=Config.MAX_SEQ_LEN)
    parser.add_argument("--output_dir", default=Config.OUTPUT_DIR)
    
    args = parser.parse_args()
    main(args)
