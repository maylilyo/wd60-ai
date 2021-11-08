import os
from pathlib import Path

import cv2
import torch
import imageio
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import helper.loss as c_loss

# ORIGIN_VIDEO -> 1부터 1199까지 ORIGINAL
# ORIGIN_30 VIDEO -> 1, 3, 5, 7, ..., 1199까지
# demo -> 1, 2(Inference), 3, 4(Inference), ..., 1199까지

# SSIM을 위해 비교해야 하는 frame = INDEX가 동일한 ORIGIN_VIDEO의 짝수 프레임과 demo의 짝수 프레임


class CenterCropper(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2, :]


PROJECT_DIR = Path(__file__).absolute().parent
DATA_DIR = PROJECT_DIR / 'data'
DEMO_VIDEO_DIR = DATA_DIR / 'demo'
#ORIGIN_VIDEO_DIR = DATA_DIR / 'origin'
ORIGIN_VIDEO_DIR = DATA_DIR / 'netflix' / 'DrivingPOV'

file_len = len(os.listdir(ORIGIN_VIDEO_DIR))

sample_path = ORIGIN_VIDEO_DIR / '1.png'
sample_frame = imageio.imread(sample_path)
cropper = CenterCropper(sample_frame.shape[:2], (256, 448))
ssim_score = c_loss.SSIM()


def preprocess_image(frame):
    frame = cropper(frame)
    frame = frame.transpose(2, 0, 1).astype(np.float32)
    frame = torch.from_numpy(frame)
    frame *= (1.0 / 255.0)
    frame = torch.unsqueeze(frame, 0)

    print(frame.shape)
    return frame


def base_preprocess_image(frame):
    frame = frame.transpose(2, 0, 1).astype(np.float32)
    frame = torch.from_numpy(frame)
    frame *= (1.0 / 255.0)
    frame = torch.unsqueeze(frame, 0)
    print(frame.shape)
    return frame


def image_score(origin_index, compare_index):
    origin_image = imageio.imread(
        ORIGIN_VIDEO_DIR / f'{origin_index}.png')
    origin_image = preprocess_image(origin_image)

    compare_image = imageio.imread(
        DEMO_VIDEO_DIR / f'{compare_index:04d}.png')  # 비교
    compare_image = base_preprocess_image(compare_image)

    score = ssim_score(origin_image, compare_image)
    print(f'{compare_index:04d} index score success')

    save_image(origin_image, DATA_DIR /
               f'result_origin_{origin_index:04d}.png')
    save_image(compare_image, DATA_DIR /
               f'result_compare_{compare_index:04d}.png')

    return score


f = open(f'{DATA_DIR}/result.txt', 'w')  # with가 안전함. 수정 예정
count = 0
sum_similar = 0

for index in range(2, file_len + 1, 2):
    count += 1
    score = image_score(index, index)
    sum_similar += score
    score_form = f"{index}th Similarity : {score:.5f}\n"
    f.write(score_form)

print(f"Average SSIM Score : {(sum_similar / count):.5f}\n")
f.close()
