import os
import sqlite3

import numpy as np
import pandas as pd
from face_detector import FaceDetector
from face_encoder import FaceEncoder
from face_dataset_encoder import encode_image


def bytes_to_float32(x):
    return np.frombuffer(x, dtype='float32')


class FacialRecogniser:
    def __init__(self, face_dir='references', n_faces=1, face_size=(224, 224),
                 threshold=.4, align=True, mtcnn=False, sqlite=True):
        self.n_faces = n_faces
        self.face_size = face_size
        self.threshold = threshold
        self.face_detector = FaceDetector(n_faces=self.n_faces, face_size=self.face_size, align=align, mtcnn=mtcnn)
        self.face_encoder = FaceEncoder()
        self.sqlite = sqlite
        self.face_dir = face_dir
        self.reference_faces = self.load_reference_faces(self.face_dir)

    def recognise_faces(self, img, return_face_img=True):
        faces = self.face_detector.detect_faces(img)
        if len(faces) < 1:
            return []
        encoded_faces = [self.face_encoder.encode(face['image']) for face in faces]
        result = []
        for face, encoded_face in zip(faces, encoded_faces):
            recognised_person = (None, float('inf'))
            if self.sqlite:
                for idx in range(len(self.reference_faces)):
                    ref = self.reference_faces.iloc[idx]['embedding']
                    person = self.reference_faces.iloc[idx]['name']
                    distance = np.linalg.norm(encoded_face - ref)
                    recognised_person = self.update_recognition(distance, person, recognised_person)
            else:
                for person in self.reference_faces:
                    reference_embeddings = self.reference_faces[person]
                    distance = np.min(([np.linalg.norm(encoded_face - ref) for ref in reference_embeddings]))
                    recognised_person = self.update_recognition(distance, person, recognised_person)
            if not return_face_img:
                face['image'] = None
            result.append({'face': face,
                           'recognised': recognised_person[0] is not None,
                           'name': recognised_person[0]})
        return result

    def load_reference_faces(self, directory):
        if self.sqlite and os.path.exists('face_embeddings.db'):
            connection = sqlite3.connect('face_embeddings.db')
            encoded_reference_faces = pd.read_sql('SELECT * FROM face_embeddings', con=connection)
            encoded_reference_faces['embedding'] = encoded_reference_faces['embedding'].apply(bytes_to_float32)
        else:
            people = os.listdir(directory)
            encoded_reference_faces = {}
            for person in people:
                person_directory = directory + '/' + person
                files = [os.path.join(person_directory, f) for f in os.listdir(person_directory) if '.jpg' in f]
                encoded_reference_faces[person] = encode_image(files, self.face_detector, self.face_encoder)
        return encoded_reference_faces

    def update_recognition(self, distance, current_name, old_result):
        if distance <= self.threshold and distance < old_result[1]:
            return current_name, distance
        return old_result


if __name__ == '__main__':
    recogniser = FacialRecogniser()
    print(recogniser.reference_faces)
