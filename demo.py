# Standard
import os
from pathlib import Path
from tqdm import tqdm

# PIP
import torch
import numpy as np
import imageio
from torchvision.utils import save_image

# Custom
from config import Config
from softcode.main_net import Main_net


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


MODEL_ID = '1634091753'

PROJECT_DIR = Path(__file__).absolute().parent
DATA_DIR = PROJECT_DIR / 'data'
ORIGIN_DIR = DATA_DIR / 'netflix' / 'DrivingPOV'
DEMO_VIDEO_DIR = DATA_DIR / 'demo'
ORIGIN_VIDEO_DIR = DATA_DIR / 'origin'
ORIGIN30_VIDEO_DIR = DATA_DIR / 'origin30'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Create demo directory
createDirectory(DEMO_VIDEO_DIR)
createDirectory(ORIGIN_VIDEO_DIR)
createDirectory(ORIGIN30_VIDEO_DIR)

# model, json 불러오기
cfg = Config()
cfg.load_options(MODEL_ID)
model = Main_net(cfg.model_option)
state_dict = torch.load(OUTPUT_DIR / f'{MODEL_ID}.pt')
model.load_state_dict(state_dict)
model.cuda()
model.eval()

# 사진 불러오기


class CenterCropper(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2, :]


sample_path = ORIGIN_DIR / '1.png'
sample_frame = imageio.imread(sample_path)
H, W = sample_frame.shape[:2]
cropper = CenterCropper(sample_frame.shape[:2], (256, 448))

file_len = len(os.listdir(ORIGIN_DIR))
frame_list = []
for origin_idx in range(1, file_len + 1):
    origin_path = ORIGIN_DIR / f'{origin_idx}.png'
    origin_frame = imageio.imread(origin_path)
    H, W = origin_frame.shape[:2]

    origin_frame = cropper(origin_frame)
    origin_frame = origin_frame.transpose(2, 0, 1).astype(np.float32)
    origin_frame = torch.from_numpy(origin_frame)
    origin_frame *= (1.0 / 255.0)
    origin_frame = torch.unsqueeze(origin_frame, 0)
    origin_frame = origin_frame.cuda()

    save_image(origin_frame, ORIGIN_VIDEO_DIR / f'{origin_idx:04d}.png')

    if origin_idx % 2 == 1:
        frame_list.append(origin_frame)
        save_image(origin_frame, ORIGIN30_VIDEO_DIR /
                   f'{origin_idx//2+1:04d}.png')

print('FINISH: load original frames')

# 불러온 모델로 추론
for idx in tqdm(range(len(frame_list) - 1)):
    front_frame = frame_list[idx]
    back_frame = frame_list[idx+1]
    inferenced_frame = model(front_frame, back_frame)

    # BGR → RGB
    inferenced_frame = torch.flip(inferenced_frame, [0])
    save_image(front_frame, DEMO_VIDEO_DIR / f'{idx * 2:04d}.png')
    save_image(inferenced_frame, DEMO_VIDEO_DIR / f'{idx * 2 + 1:04d}.png')
