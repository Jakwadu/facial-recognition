import io
import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from facial_recogniser import FacialRecogniser

DEFAULT_REFERENCE_FACE_PATH = 'references'
SIMILARITY_THRESHOLD = 0.6

app = FastAPI()
recogniser = FacialRecogniser(DEFAULT_REFERENCE_FACE_PATH, threshold=SIMILARITY_THRESHOLD)


@app.get('/')
def index():
    return {
        'name': 'Facial recognition System',
        'encoder_type': recogniser.face_encoder.encoder_type
    }


@app.get('/config')
def get_configuration():
    return {
        'encoder_type': recogniser.face_encoder.encoder_type,
        'similarity_threshold': recogniser.threshold,
        'reference_image_directory': recogniser.face_dir
    }


@app.post('/recognise')
async def run_facial_recognition(req: Request):
    data = await req.body()
    img = np.load(io.BytesIO(data), allow_pickle=True)
    result = recogniser.recognise_faces(img, return_face_img=False)
    if any([r['recognised'] is True for r in result]):
        print(result)
    return {'result': result}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
