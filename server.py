from fastapi import FastAPI
from facial_recogniser import FacialRecogniser
from facial_recognition_utils import recognise_face_in_image, recognise_face_in_video_stream

DEFAULT_REFERENCE_FACE_PATH = 'references'
ENCODER_TYPE = 'vgg-face'
SIMILARITY_THRESHOLD = 0.4

app = FastAPI()
recogniser = FacialRecogniser(ENCODER_TYPE, DEFAULT_REFERENCE_FACE_PATH, threshold=SIMILARITY_THRESHOLD)


@app.get('/')
def index():
    return {
        'name': 'Facial recognition System',
        'encoder_type': recogniser.face_encoder.encoder_type}


@app.get('/config')
def get_configuration():
    config = {
        'encoder_type': recogniser.face_encoder.encoder_type,
        'similarity_threshold': recogniser.threshold,
        'reference_image_directory': recogniser.face_dir
    }
    return config


@app.post('/recognise/{img}')
def run_facial_recognition(img):
    return recogniser.recognise_faces(img)