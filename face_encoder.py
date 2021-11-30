import numpy as np
from vgg_face import build_encoder


class FaceEncoder:
    def __init__(self):
        self.encoder = build_encoder()

    def encode(self, img):
        if len(img.shape) == 3:
            img = np.array([img])
        # Ensure values are between -1 and 1
        preprocessed_img = (img / np.max(img)) * 2 - 1
        return self.encoder(preprocessed_img)
