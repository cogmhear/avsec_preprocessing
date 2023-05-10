# Scripts for preprocessing audio-visual speech enhancement challenge (AVSEC) data

### This script can be used to extract the following features
- FaceMesh landmarks [1] 
- lip images using landmark
- face embeddings using FaceNet [2]
- lip embeddings using TCN [3]

## Requirements

```text
## CPU 
pip install -r requirements.txt

## GPU
pip install -r requirements_gpu.txt

## Apple Silicon
pip install -r requirements_mac.txt
```

## Usage
```bash
python main.py --data-dir ./data/train/scenes \
               --save-dir ./preprocessed/train \
               --models-root ./models \
               --all-feat
```

## References

- [1] [MediaPipe Face Mesh](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/)
- [2] [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet)
- [3] [Lipreading using Temporal Convolutional Networks](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)