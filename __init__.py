import numpy as np

# Local Imports
from src.gan import GAN
from src.utils import load_frames
import config

if __name__ == '__main__':
    imgs = load_frames(n_samples=20, source_folder=config.source_folder)
    
    gan = GAN(config)
    gan.plot_examples(imgs)
    gan.fit()