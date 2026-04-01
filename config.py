import torch
import os

class Config:
    # Data Paths
    TRAIN_CSV = "train.csv"
    OUTPUT_DIR = "./outputs"

    # Feature Definitions
    CAT_FEATURES = [
        'sex',
        'numberGame',
        'strikeId',
        'handId',
        'strengthId',
        'spinId',
        'positionId',
        'actionId',
        'pointId',
        'phase_id',
        'action_group',
        'is_odd_stroke',
        'prev_actionId',
        'prev_pointId',
        'prev_spinId',
        'prev_handId',
        'prev_positionId',
    ]

    NUM_FEATURES = [
        'strikeNumber',
        'scoreSelf',
        'scoreOther',
        'scoreDiff',
        'absScoreDiff',
        'is_close_score',
        'is_game_point_like',
        'log_strike',
        'is_late_rally',
        'recent_atk_density',

        'recent_control_density',
        'recent_action_group_change_rate',
        'strike_bucket'
    ]

    # Model Hyperparameters
    MAX_SEQ_LEN = 8
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 256
    DROPOUT = 0.2

    # Training Hyperparameters
    BATCH_SIZE = 256
    EPOCHS = 50
    LR = 2e-4
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

