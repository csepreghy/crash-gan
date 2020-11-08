import numpy as np

# Local Imports
from gan import GAN
from utils import load_frames
import config

if __name__ == '__main__':
    imgs = load_frames()
    gan = GAN(config)
    gan.plot_examples(imgs)
    print(f'imgs.shape = {imgs.shape}')