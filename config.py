import torch
import os

class Config:
    # Data Paths
    TRAIN_CSV = "train.csv"
    OUTPUT_DIR = "./outputs"
    
    # Feature Definitions
    CAT_FEATURES = [
        'sex', 'numberGame', 'gamePlayerId', 'gamePlayerOtherId', 
        'strikeId', 'handId', 'strengthId', 'spinId', 
        'positionId', 'actionId', 'pointId', 'phase_id',
        'striker_id', 'is_server'
    ]
    NUM_FEATURES = [
        'strikeNumber', 'scoreSelf', 'scoreOther', 'scoreDiff',
        'is_deuce', 'is_close_score', 'is_third_stroke'
    ]
    
    # Model Hyperparameters
    MAX_SEQ_LEN = 15
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 256
    DROPOUT = 0.2
    
    # Training Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    EARLY_STOPPING_PATIENCE = 7
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # Task Weights
    LOSS_WEIGHTS = {
        'action': 0.4,
        'point': 0.4,
        'outcome': 0.2
    }
