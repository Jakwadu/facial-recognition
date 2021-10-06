import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from matplotlib import pyplot as plt


class FaceDetector:
    def __init__(self, n_faces=1, face_size=None, mtcnn=False):
        self.n_faces = n_faces
        if face_size is not None:
            assert isinstance(face_size, (tuple, list)), 'face_size must be a tuple or list'
            assert len(face_size) == 2, 'face_size must be of length 2'
        self.face_size = face_size
        self.mtcnn = mtcnn
        if self.mtcnn:
            self.detector = MTCNN()
        else:
            self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        if self.mtcnn:
            result = self.detector.detect_faces(frame)
        else:
            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            result = self.detector.detectMultiScale(grey, 1.3, 5)
        faces = []
        for idx in range(len(result[:self.n_faces])):
            if self.mtcnn:
                x0, y0, w, h = result[idx]['box']
                x0, y0 = abs(x0), abs(y0)
            else:
                x0, y0, w, h = result[idx]
            x1 = x0 + w
            y1 = y0 + h
            face = frame[y0:y1, x0:x1]
            if self.face_size is not None:
                face = Image.fromarray(face)
                face = face.resize(self.face_size)
            faces.append({'image': np.asarray(face), 'co-ordinates': (x0, y0, x1, y1)})
        return faces


if __name__ == "__main__":
    detector = FaceDetector()
    img = plt.imread('target_faces/sources/GettyImages-1228393166.jpg')
    face = detector.detect_faces(img)[0]['image']
    plt.imshow(face)
    plt.show()
