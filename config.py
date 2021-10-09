# Standard
from pathlib import Path

# PIP
import pytorch_lightning as pl

# Custom


class Config:
    # User Setting
    SEED = 42

    # Path
    PROJECT_DIR = Path(__file__).absolute().parent
    DATA_DIR = PROJECT_DIR / 'data'
    VIMEO_DIR = DATA_DIR / 'vimeo_triplet'
    WEIGHT_DIR = PROJECT_DIR / 'weights'

    # Training
    GPUS = [0, 1, 2, 3]
    MAX_EPOCHS = 20
    EARLYSTOP_PATIENCE = 10
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    CRITERION = 'LapLoss'
    OPTIMIZER = 'Adam'
    LR_SCHEDULER = 'StepLR'

    # Model
    IS_CROP = False

    # Dataset
    SEQ_LEN = 10

    # Log
    PROJECT_TITLE = 'Softsplat'
    WANDB_NAME = 'pwcnet'
    IS_PROGRESS_LOG_ON = True

    def __init__(self, seed=None):
        self.set_random_seed(seed)
        self.set_model_option()

    def set_random_seed(self, seed):
        if seed:
            self.SEED = seed

        pl.seed_everything(self.SEED)

    def set_model_option(self):
        if self.IS_CROP:
            width = 256
            height = 256
        else:
            width = 448
            height = 256

        self.model_option = {
            'shape': [height, width],
        }
