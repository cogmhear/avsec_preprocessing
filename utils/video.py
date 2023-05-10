import os

import imageio
import cv2


def get_frames_from_video(vid_path, rgb=True, imsize=None):
    assert os.path.isfile(vid_path), "Error loading video. File provided does not exist, cannot read: {}".format(
        vid_path)
    vidcap = cv2.VideoCapture(vid_path)
    image_files = []
    success, image = vidcap.read()
    while success:
        if rgb:
            image_new = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_new = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if imsize is not None:
            image_new = cv2.resize(image_new, imsize)
        image_files.append(image_new)
        success, image = vidcap.read()
    return image_files


def write_video_mp4(out_path, frames, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (96, 96))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
