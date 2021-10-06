import os
import numpy as np
from face_detector import FaceDetector
from face_encoder import FaceEncoder
from face_dataset_encoder import encode_image


class FacialRecogniser:
    def __init__(self, face_dir, encoder_type, n_faces=1, face_size=(299, 299), threshold=.4):
        self.n_faces = n_faces
        self.face_size = face_size
        self.threshold = threshold
        self.face_detector = FaceDetector(n_faces=self.n_faces, face_size=self.face_size)
        self.face_encoder = FaceEncoder(encoder_type=encoder_type)
        self.face_dir = face_dir
        self.reference_faces = self.load_reference_faces(self.face_dir)

    def recognise_faces(self, img):
        faces = self.face_detector.detect_faces(img)
        if len(faces) < 1:
            return []
        encoded_faces = [self.face_encoder.encode(face['image']) for face in faces]
        distances = [[np.linalg.norm(encoded_face - ref) for ref in self.reference_faces] for encoded_face in encoded_faces]
        print('Distances:', distances)
        recognised = [any([dim < self.threshold for dim in dist]) for dist in distances]
        print(self.threshold)
        print(recognised)
        return [{'face': f, 'recognised': r} for f, r in zip(faces, recognised)]

    def load_reference_faces(self, directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
        encoded_reference_faces = encode_image(files, self.face_detector, self.face_encoder)
        # return np.mean(encoded_reference_faces, axis=0)
        return encoded_reference_faces
