import numpy as np
import torch
import cv2
from torchvision.utils import save_image
from icecream import ic

from main_net import Main_net
from test import save_test_image


def read_sample_images():
    img1 = cv2.imread(filename='./pair_img/im1.png', flags=-1)
    img2 = cv2.imread('./pair_img/im2.png')
    img3 = cv2.imread(filename='./pair_img/im3.png', flags=-1)

    return img1, img2, img3


def normalize_image(img):
    out = img.transpose(2, 0, 1)
    out = out[None, :, :, :]
    out = out.astype(np.float32)
    out *= (1.0 / 255.0)

    out = np.ascontiguousarray(out)
    out = torch.FloatTensor(out)

    return out


height = 256
width = 448
shape = [1, 3, height, width]

img1, img2, img3 = read_sample_images()

tenFirst = normalize_image(img1)
tenFirst = tenFirst.cuda()

tenSecond = normalize_image(img3)
tenSecond = tenSecond.cuda()

model = Main_net(shape)  # nn.Module
model.load_state_dict(torch.load('weights/softsplat.pt'))
model.eval()  # layer의 동작을 inference로 바꿔줌. no_grad와 함께 사용한다.
model = model.cuda()

with torch.no_grad():
    # inference를 진행할 때 주로 사용하는 코드. 사용하지 않는 gradient를 끔으로써 연산 속도 증가

    # tenFirst, tenSecond: (1, 3, height, width)
    img_out = model(tenFirst, tenSecond)
    # img_out: (1, 3, height, width)

    img_out = img_out.squeeze()
    # img_out: (3, height, width)

save_test_image(img_out, 'img_out.jpg')
