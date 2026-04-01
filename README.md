# Table Tennis Tactics & Outcome Prediction

本專案用於「基於時序資料之桌球戰術與結果預測競賽」。使用 PyTorch 實作 Multi-Task Transformer Encoder 模型，預測下一拍球種、落點及 Rally 結果。

## 專案結構
- `data/`: 資料工程與 Dataset 定義
- `models/`: Transformer 模型組件
- `train/`: 損失函數、指標計算與訓練器
- `utils/`: 通用工具 (如隨機種子設定)
- `main.py`: 訓練入口程式
- `config.py`: 超參數與特徵定義

## 資料格式假設
輸入 CSV 需包含：
- `rally_uid`: 每一回合成對唯一識別碼
- `match`: 比賽識別碼 (用於 Group Split)
- `strikeNumber`: 拍數
- `serverGetPoint`: 該回合最終得分者 (0/1)
- `actionId`, `pointId`: 類別特徵
- 其他戰術特徵如 `sex`, `handId`, `spinId` 等

## 如何訓練
1. 安裝依賴：`pip install -r requirements.txt`
2. 放置資料：確保 `train.csv` 在根目錄或指定路徑
3. 執行訓練：
   ```bash
   python main.py --train_csv train.csv --epochs 50 --batch_size 64
   ```

## 模型預測
模型會根據前 $n-1$ 拍預測第 $n$ 拍。
預測目標：
1. `actionId`: Multi-class
2. `pointId`: Multi-class
3. `serverGetPoint`: Binary (Rally Outcome)

## 修改特徵
- 若要增加類別特徵，請修改 `config.py` 中的 `CAT_FEATURES`。
- 若要增加數值特徵，請在 `data/features.py` 中實作計算邏輯，並加入 `Config.NUM_FEATURES`。
