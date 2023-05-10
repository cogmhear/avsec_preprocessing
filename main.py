import os
import shutil
from os import listdir, makedirs
from os.path import isdir, join, isfile

import gdown
import onnxruntime as ort
from clize import run
from tqdm import tqdm

from utils.generic import check_flags
from utils.landmark import *
from utils.lips import get_lip_images
from utils.video import get_frames_from_video, write_video_mp4


def normalise_lips(lips_arr):
    lips_arr = lips_arr.astype(np.float32)
    lips_arr /= 255
    lips_arr = (lips_arr - 0.421) / 0.165
    return lips_arr


def preprocess(*, data_dir, save_dir, models_root, vid_ext="mp4", all_feat=False, landmark=False,
               lip_images=False, lip_embed=False,
               face_embed=False, gpu=False):
    """
    :param data_dir: path to scenes root e.g. data/train/scenes
    :param save_dir: path to save preprocessed features
    :param models_root: pretrained models path for downloading
    :param vid_ext: videos with vid_ext will be processed
    :param all_feat: Flag to extract all supported features
    :param landmark: Flag to extract landmark features
    :param lip_images: Flag to extract lip images
    :param face_embed: Flag to extract face_embed
    :param lip_embed: Flag to extract lip_embed (requires lip_images=True)
    :param gpu: Flag to use GPU
    """
    assert isdir(data_dir), "Data root does not exists"
    makedirs(save_dir, exist_ok=True)
    face_embed, landmark, lip_embed, lip_images = check_flags(all_feat, face_embed, landmark, lip_embed, lip_images)
    videos_list = [file for file in listdir(data_dir) if file.endswith("_silent.{}".format(vid_ext))]
    landmark_root = join(save_dir, "facemesh_landmark")
    makedirs(landmark_root, exist_ok=True)
    lips_root = join(save_dir, "lips")
    makedirs(lips_root, exist_ok=True)
    lip_embedding = join(save_dir, "lip_embedding")
    makedirs(lip_embedding, exist_ok=True)
    face_embedding = join(save_dir, "face_embedding")
    makedirs(face_embedding, exist_ok=True)
    if not isdir(models_root):
        os.makedirs(models_root)
        gdown.download(id="1jljUlSyYSpGy9FCTlzEiRptY8eyXEcFj", output=join(models_root, "model.zip"))
        shutil.unpack_archive(join(models_root, "model.zip"), models_root)
    if gpu:
        execution_provider = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        execution_provider = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    if landmark:
        mediapipe_facemesh = mediapipe_facemesh_model()
    if lip_embed:
        lip_embed_model = ort.InferenceSession(join(models_root, "lrw_resnet18_dctcn.onnx"),
                                               providers=execution_provider)
    if face_embed:
        face_embed_model = ort.InferenceSession(join(models_root, "facenet_embed.onnx"),
                                                providers=execution_provider)

    for video in tqdm(videos_list):
        landmark_file = join(landmark_root, video.replace(f".{vid_ext}", ".npz"))
        faceembed_file = join(face_embedding, video.replace(f".{vid_ext}", ".npz"))
        lipembed_file = join(lip_embedding, video.replace(f".{vid_ext}", ".npz"))
        lip_file = join(lips_root, video)
        if face_embed and not isfile(faceembed_file):
            face_frames = get_frames_from_video(join(data_dir, video), imsize=(160, 160))
            face_embedding_npz_list = []
            for i in range(0, len(face_frames), 100):
                face_embedding_npz = face_embed_model.run(None, {"input": face_frames[i:i + 100]})[0]
                face_embedding_npz_list.append(face_embedding_npz)
            face_embedding_npz = np.concatenate(face_embedding_npz_list, axis=0)
            np.savez(faceembed_file, data=face_embedding_npz)

        if landmark and not isfile(landmark_file):
            video_frames = get_frames_from_video(join(data_dir, video))
            facemesh_landmark = []
            for image in video_frames:
                try:
                    landmark_feat = mediapipe_get_landmark(image, mediapipe_facemesh)
                    facemesh_landmark.append(landmark_feat)
                except Exception as e:
                    if len(facemesh_landmark) > 0:
                        landmark_feat = facemesh_landmark[-1]
                        facemesh_landmark.append(landmark_feat)
                    else:
                        print("Unable to extract landmark features for, {} : {}".format(video, e))
            if len(facemesh_landmark) != len(video_frames):
                if len(facemesh_landmark) == 0:
                    with open("§", "a") as f:
                        f.write(f"{landmark_file}\n")
                    continue

                padding_frames = len(video_frames) - len(facemesh_landmark)
                for _ in range(padding_frames):
                    facemesh_landmark.append(facemesh_landmark[-1])
            landmark_feat = np.array(facemesh_landmark, dtype=np.float16)
            np.savez(landmark_file, data=landmark_feat)
        else:
            landmark_feat = None
            video_frames = None

        if lip_images and not isfile(lip_file):
            landmark_feat = np.load(landmark_file)["data"] if landmark_feat is None else landmark_feat
            frames = get_frames_from_video(join(data_dir, video)) if video_frames is None else video_frames
            try:
                lips = get_lip_images(frames, landmark_feat)
                write_video_mp4(lip_file, lips)
            except Exception as e:
                print("Unable to extract lip images for {}:{}".format(video, e))
        if lip_embed and isfile(lip_file) and not isfile(lipembed_file):
            lip_img = get_frames_from_video(lip_file, rgb=False, imsize=(88, 88))
            lip_embedding_npz_list = []
            for i in range(0, len(lip_img), 100):
                lips = normalise_lips(np.array(lip_img[i:i + 100]))
                lip_embedding_npz = lip_embed_model.run(None, {"input": lips[np.newaxis, np.newaxis, ...]})[0][0]
                lip_embedding_npz_list.append(lip_embedding_npz)
            lip_embedding_npz = np.concatenate(lip_embedding_npz_list, axis=0)
            np.savez(lipembed_file, data=lip_embedding_npz)


if __name__ == "__main__":
    run(preprocess)
