# Standard library imports
import numpy as np

# Local application imports
from src.gan import GAN
from src.utils import load_frames
import src.config as config

if __name__ == '__main__':
    imgs = load_frames(n_samples=20, source_folder=config.source_folder)
    
    gan = GAN(config)
    gan.plot_examples(imgs)
    gan.fit()