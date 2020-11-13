import subprocess
import cv2
import glob
import numpy as np
import random

def remove_audio():
    command = 'for file in data/*.mp4; do ffmpeg -i "$file" -c copy -an "$file"; done'
    subprocess.call(command, shell=True)

def get_all_frames():
    vidcap = cv2.VideoCapture('data/test/crash360p_1024k.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("data/test/frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def load_frames(n_samples, source_folder):
    filenames = glob.glob(f'{source_folder}/*.jpg')
    random.shuffle(filenames)

    imgs = []
    for filename in filenames[0:n_samples]:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_square = resize(img, (128,128), 0)
        imgs.append(img_square)
    
    return np.array(imgs)

def resize(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    new_w = sw
    new_h = np.round(new_w/aspect).astype(int)
    pad_vert = (sh-new_h)/2
    pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
    pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

if __name__ == '__main__':
    imgs = load_frames()
    print(f'imgs.shape = {imgs.shape}')
    # get_all_frames()
    # remove_audio()

