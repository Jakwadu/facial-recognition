import os
import numpy as np
from face_detector import FaceDetector
from face_encoder import FaceEncoder
from face_dataset_encoder import encode_image


class FacialRecogniser:
    def __init__(self, encoder_type, face_dir='references', n_faces=1,
                 face_size=(224, 224), threshold=.4, align=True, mtcnn=False):
        self.n_faces = n_faces
        self.face_size = face_size
        self.threshold = threshold
        self.face_detector = FaceDetector(n_faces=self.n_faces, face_size=self.face_size, align=align, mtcnn=mtcnn)
        self.face_encoder = FaceEncoder(encoder_type=encoder_type)
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
            for person in self.reference_faces:
                reference_embeddings = self.reference_faces[person]
                distance = np.min(([np.linalg.norm(encoded_face - ref) for ref in reference_embeddings]))
                if distance <= self.threshold and distance < recognised_person[1]:
                    recognised_person = (person, distance)
            if not return_face_img:
                face['image'] = None
            result.append({'face': face,
                           'recognised': recognised_person[0] is not None,
                           'name': recognised_person[0]})
        return result

    def load_reference_faces(self, directory):
        people = os.listdir(directory)
        encoded_reference_faces = {}
        for person in people:
            person_directory = directory + '/' + person
            files = [os.path.join(person_directory, f) for f in os.listdir(person_directory) if '.jpg' in f]
            encoded_reference_faces[person] = encode_image(files, self.face_detector, self.face_encoder)
        return encoded_reference_faces


if __name__ == '__main__':
    recogniser = FacialRecogniser('vgg-face')
    print(recogniser.reference_faces)
