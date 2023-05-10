import numpy as np
import mediapipe as mp


def mediapipe_facemesh_model(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.3,
                             refine_landmarks=True):
    mp_face_mesh = mp.solutions.face_mesh
    mediapipe_facemesh = mp_face_mesh.FaceMesh(static_image_mode=static_image_mode, max_num_faces=max_num_faces,
                                               refine_landmarks=refine_landmarks,
                                               min_detection_confidence=min_detection_confidence)
    return mediapipe_facemesh


def mediapipe_get_landmark(image, mediapipe_facemesh=None):
    if mediapipe_facemesh is None:
        mediapipe_facemesh = mediapipe_facemesh_model()
    results = mediapipe_facemesh.process(image)
    for face_landmarks in results.multi_face_landmarks:
        shape_np = np.array(
            [(data_point.x, data_point.y, data_point.z) for data_point in face_landmarks.landmark])
        return shape_np * np.array([image.shape[1], image.shape[0], image.shape[2]])
