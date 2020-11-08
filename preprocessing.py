import subprocess
import cv2

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


if __name__ == '__main__':
    get_all_frames()
    # remove_audio()

