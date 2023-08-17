import torch

from dataset.DIV2K import DIV2K

from model import Generator

import torchvision.transforms as T
from PIL import Image

if __name__ == "__main__":

    MODEL_PATH = 'runs/models/gen_latest.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = '/Users/muhammadarham/Drive/MLProjects/SRGAN/data/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR'

    test_transform = T.Compose([
        T.ToTensor(),
        # T.Normalize(mean=[0., 0., 0.], std=[1, 1, 1])
    ])

    img_hr_path = '/Users/muhammadarham/Drive/MLProjects/SRGAN/data/DIV2K/DIV2K_train_HR/DIV2K_train_HR/0001.png'
    img_hr = Image.open(img_hr_path).resize((48, 48))

    low_res = test_transform(img_hr)

    gen = Generator(in_channels=3, num_channels=64, num_blocks=16)
    gen.load_state_dict(torch.load(MODEL_PATH))
    gen.eval().to(DEVICE)
    sr = gen(low_res.unsqueeze(axis=0)).squeeze(axis=0)

    transform = T.ToPILImage()
    sr = transform(sr)
    # hr = transform(high_res)

    sr.save('runs/sr.png')
    img_hr.save('runs/hr.png')
