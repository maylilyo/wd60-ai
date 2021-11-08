# Standard
from pathlib import Path
import json
import time

# PIP
import torch
import pytorch_lightning as pl

# Custom


class Config:
    # User Setting
    SEED = 100

    # Path
    PROJECT_DIR = Path(__file__).absolute().parent
    DATA_DIR = PROJECT_DIR / 'data'
    VIMEO_DIR = DATA_DIR / 'vimeo_triplet'
    WEIGHT_DIR = PROJECT_DIR / 'weights'
    OUTPUT_DIR = PROJECT_DIR / 'output'

    # Training
    GPUS = [0, 1, 2, 3]
    MAX_EPOCHS = 10
    EARLYSTOP_PATIENCE = 100
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-6
    CRITERION = 'LapLoss'
    OPTIMIZER = 'AdamW'
    LR_SCHEDULER = 'StepLR'

    # Model
    IS_CROP = False
    FLOW_EXTRACTOR = 'ifnet'  # pwcnet, ifnet
    IS_FREEZE = False

    # Dataset
    NUM_WORKERS = 8

    # Log
    MODEL_ID = str(int(time.time()))
    PROJECT_TITLE = 'Softsplat'
    IS_PROGRESS_LOG_ON = True

    def __init__(self, seed=None):
        Config.BATCH_SIZE *= len(Config.GPUS)

        self.set_random_seed(seed)
        self.set_model_option()

    def set_random_seed(self, seed):
        if seed:
            self.SEED = seed

        pl.seed_everything(self.SEED)
        torch.backends.cudnn.benchmark = False

    def set_model_option(self):
        if self.IS_CROP:
            width = 256
            height = 256
        else:
            width = 448
            height = 256

        self.model_option = {
            'shape': [height, width],
            'flow_extractor': self.FLOW_EXTRACTOR,
            'is_freeze': self.IS_FREEZE,
        }

    def save_options(self, additional_log):
        # Set option dict
        option_dict = {
            'is_crop': self.IS_CROP,
            'flow_extractor': self.FLOW_EXTRACTOR,
            'is_freeze': self.IS_FREEZE,
            'train_option': {
                'max_epochs': self.MAX_EPOCHS,
                'seed': self.SEED,
                'gpus': len(self.GPUS),
                'batch_size': self.BATCH_SIZE,
                'learning_rate': self.LEARNING_RATE,
            },
            'additional_log': additional_log,
        }

        # Save dict to json
        with open(self.OUTPUT_DIR / f'{self.MODEL_ID}.json', 'w') as json_file:
            json.dump(option_dict, json_file, indent=2)

    def load_options(self, model_id):
        # Load dict from json
        with open(self.OUTPUT_DIR / f'{model_id}.json', 'r') as json_file:
            option_dict = json.load(json_file)

        # Overwrite options
        self.IS_CROP = option_dict['is_crop']
        self.FLOW_EXTRACTOR = option_dict['flow_extractor']
        self.IS_FREEZE = option_dict['is_freeze']

        self.set_model_option()
