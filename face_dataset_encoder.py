import os
import sys
import pickle
import tensorflow as tf
import numpy as np
from face_encoder import FaceEncoder
from face_detector import FaceDetector
from tqdm import tqdm

img_dir = 'target_faces/sources'
face_source_dir = 'c:/Users/Jamie/Documents/img_align_celeba'
output_file = 'target_faces/target_face_encodings'
encode = False
batch_size = 128
encoding_file = os.path.realpath(output_file+'.pkl')
encoder_type = 'xception'
saved_faces_directory = 'faces'


def encode_image(img_paths, face_detector, img_encoder):
    imgs = list(map(tf.keras.preprocessing.image.load_img, img_paths))
    imgs = [np.asarray(img) for img in imgs]
    faces = [np.array([face['image'] for face in face_detector.detect_faces(img)]) for img in imgs]
    faces = np.concatenate([face for face in faces if len(face.shape) == 4])
    face_encodings = img_encoder.encode(faces)
    return face_encodings


def save_faces(directory, face_detector):
    files = [os.path.join(directory, f_) for f_ in os.listdir(directory)]
    idx = 0
    for f_ in tqdm(files, desc='Saving faces detected in images'):
        img = np.asarray(tf.keras.preprocessing.image.load_img(f_))
        faces = [face['image'] for face in face_detector.detect_faces(img)]
        for face in faces:
            if len(face) > 0:
                tf.keras.preprocessing.image.save_img(os.path.join(saved_faces_directory, f'{idx}.jpg'), face)
                idx += 1


def read_and_encode_images(directory, face_detector, img_encoder):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    encodings = []
    for idx in tqdm(range(0, len(files), batch_size), desc='Encoding image batches'):
        encodings.append(encode_image(files[idx:idx+batch_size], face_detector, img_encoder))
    encodings = np.concatenate(encodings)
    return encodings


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
    detector = FaceDetector(1, face_size=(128, 128))
    if encode:
        encoder = FaceEncoder(encoder_type=encoder_type)
        encoded_faces = read_and_encode_images(img_dir, detector, encoder)
        with open(encoding_file, 'wb') as f:
            pickle.dump(encoded_faces, f)
    else:
        save_faces(face_source_dir, detector)
