import subprocess
import cv2
import glob
import numpy as np

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

def load_frames():
    filenames = glob.glob("data/test/frames/*.jpg")
    imgs = []
    for filename in filenames[0:10]:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    
    return np.array(imgs)
    


if __name__ == '__main__':
    imgs = load_frames()
    print(f'imgs.shape = {imgs.shape}')
    # get_all_frames()
    # remove_audio()

