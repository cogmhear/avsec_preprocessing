import numpy as np
import cv2


def get_lip_bbox_facemesh(frame, landmark, max_lip_dim=None):
    """Get the bounding box of the lips.
    :param frame: The frame to get the bounding box from.
    :param landmark: The landmark to get the bounding box from.
    :param max_lip_dim: The maximum dimension of the bounding box.
    :return: The bounding box of the lips.
    """
    h, w, c = frame.shape
    y_pos = landmark.astype(np.int32)[(61, 291), :][:, 0]
    x_pos = landmark.astype(np.int32)[(0, 17), :][:, 1]
    if max_lip_dim is not None:
        max_dim = max_lip_dim
    else:
        max_dim = int(max(abs(y_pos[1] - y_pos[0]), abs(x_pos[1] - x_pos[0])) * 1.1)
    center = int(sum(x_pos) / 2), int(sum(y_pos) / 2)
    left = max(0, center[0] - max_dim)
    right = min(w, center[0] + max_dim)
    top = max(0, center[1] - max_dim)
    bottom = min(h, center[1] + max_dim)
    return dict(left=left, right=right, top=top, bottom=bottom, max_dim=max_dim)


def get_lip_images(frames, facemesh_landmark, output_size=(96, 96)):
    bboxs = [get_lip_bbox_facemesh(frames[idx], facemesh_landmark[idx]) for idx in range(len(frames))]
    lip_images = []
    for idx in range(len(frames)):
        try:
            box = bboxs[idx]
            left, right, top, bottom = box["left"], box["right"], box["top"], box["bottom"]
            lip_image = frames[idx][left:right, top:bottom, :]
            if output_size is not None:
                lip_image = cv2.resize(lip_image, output_size)
            lip_images.append(lip_image)
        except:
            lip_images.append(lip_images[-1])
    return lip_images
