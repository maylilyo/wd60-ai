import torch
from torchvision.utils import save_image


def save_test_image(image, file_name='test.jpg'):

    # BGR → RGB
    image = torch.flip(image, [0])

    save_image(image, f'result/{file_name}')
