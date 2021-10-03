import os
import sys
import pickle
import tensorflow as tf
import numpy as np
from face_encoder import FaceEncoder
from face_detector import FaceDetector
from sklearn.decomposition.pca import PCA
from tqdm import tqdm

img_dir = 'target_faces/sources'
output_file = 'target_faces/target_face_encodings'
batch_size = 128
encoding_file = os.path.realpath(output_file+'.pkl')
encoder_type = 'xception'
pca = PCA(3)
use_decomposition = False
fitted_pca = os.path.realpath(f'fitted_pca_{encoder_type}.pkl')


def encode_image(img_paths, face_detector, img_encoder):
    imgs = list(map(tf.keras.preprocessing.image.load_img, img_paths))
    imgs = [np.asarray(img) for img in imgs]
    faces = [np.array(face_detector.detect_faces(img)) for img in imgs]
    faces = np.concatenate([face for face in faces if len(face.shape) == 4])
    face_encodings = img_encoder.encode(faces)
    return face_encodings


def read_and_encode_images(dir, face_detector, img_encoder):
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    encodings = []
    for idx in tqdm(range(0, len(files), batch_size), desc='Encoding image batches'):
        encodings.append(encode_image(files[idx:idx+batch_size], face_detector, img_encoder))
    encodings = np.concatenate(encodings)
    if use_decomposition:
        decomposed_encodings = pca.fit_transform(encodings)
        return decomposed_encodings
    else:
        return encodings


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
    detector = FaceDetector(1, face_size=(299, 299))
    encoder = FaceEncoder(encoder_type=encoder_type)
    encoded_faces = read_and_encode_images(img_dir, detector, encoder)
    with open(encoding_file, 'wb') as f:
        pickle.dump(encoded_faces, f)
    if use_decomposition:
        with open(fitted_pca, 'wb') as f:
            pickle.dump(pca, f)
