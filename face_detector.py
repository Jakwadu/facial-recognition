import os
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from matplotlib import pyplot as plt


class FaceDetector:
    def __init__(self, n_faces=1, face_size=None, mtcnn=False, align=True):
        self.n_faces = n_faces
        if face_size is not None:
            assert isinstance(face_size, (tuple, list)), 'face_size must be a tuple or list'
            assert len(face_size) == 2, 'face_size must be of length 2'
        self.face_size = face_size
        self.mtcnn = mtcnn
        self.align = align
        if self.mtcnn:
            self.detector = MTCNN()
        else:
            detector_config_path = os.path.join(os.path.dirname(cv2.__file__), 'data')
            self.detector = cv2.CascadeClassifier(os.path.join(detector_config_path,
                                                               'haarcascade_frontalface_default.xml'))
            self.eye_detector = cv2.CascadeClassifier(os.path.join(detector_config_path,
                                                                   'haarcascade_eye.xml'))

    def detect_faces(self, frame):
        if self.mtcnn:
            result = self.detector.detect_faces(frame)
        else:
            frame = frame.astype('uint8')
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
            if self.align:
                face = self.align_face(face)
            if self.face_size is not None:
                face = Image.fromarray(face)
                face = face.resize(self.face_size)
            faces.append({'image': np.asarray(face), 'co-ordinates': (x0, y0, x1, y1)})
        return faces

    def align_face(self, face):
        # Locate eye positions
        grey = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        eyes = self.eye_detector.detectMultiScale(grey, 1.1, 10)
        if len(eyes) >= 2:
            eye0 = eyes[0]
            eye1 = eyes[1]
            if eye0[0] < eye1[0]:
                left_eye = eye0
                right_eye = eye1
            else:
                left_eye = eye1
                right_eye = eye0
            left_eye = int(left_eye[0] + left_eye[2]/2), int(left_eye[1] + left_eye[3]/2)
            right_eye = int(right_eye[0] + right_eye[2] / 2), int(right_eye[1] + right_eye[3] / 2)

            # Rotate image to achieve alignment using the cosine rule
            left_x, left_y = left_eye
            right_x, right_y = right_eye
            if left_y > right_y:
                point_3rd = (right_x, left_y)
                direction = -1
            else:
                point_3rd = (left_x, right_y)
                direction = 1
            a = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            b = np.linalg.norm(np.array(right_eye) - np.array(point_3rd))
            c = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
            if b != 0 and c != 0:
                cos_a = (b * b + c * c - a * a) / (2 * b * c)
                angle = np.arccos(cos_a)
                angle = (angle * 180) / np.pi
                if direction == -1:
                    angle = 90 - angle
                face = Image.fromarray(face)
                face = np.array(face.rotate(direction * angle))
        return face


if __name__ == "__main__":
    detector = FaceDetector()
    img = plt.imread('references/Adelaide_Kan/14_Adelaide_Kane_0003.jpg')
    detected_face = detector.detect_faces(img)[0]['image']
    plt.imshow(detected_face)
    plt.show()
