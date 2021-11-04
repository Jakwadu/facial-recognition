import numpy as np
import tensorflow.keras.applications.xception as xception
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.applications.vgg16 as vgg16
from vgg_face import build_encoder

encoder_types = ['xception', 'resnet50', 'vgg16', 'vgg-face']


class FaceEncoder:
    def __init__(self, encoder_type='vgg-face'):
        assert encoder_type in encoder_types, 'Invalid encoder type'
        self.encoder_type = encoder_type
        if self.encoder_type == 'xception':
            self.encoder = xception.Xception(include_top=False, pooling='avg')
        elif self.encoder_type == 'resnet50':
            self.encoder = resnet50.ResNet50(include_top=False, pooling='avg')
        elif self.encoder_type == 'vgg16':
            self.encoder = vgg16.VGG16(include_top=False, pooling='avg')
        elif self.encoder_type == 'vgg-face':
            self.encoder = build_encoder()

    def encode(self, img):
        if len(img.shape) == 3:
            img = np.array([img])
        if self.encoder_type == 'xception':
            preprocessed_img = xception.preprocess_input(img)
        elif self.encoder_type == 'resnet50':
            preprocessed_img = resnet50.preprocess_input(img)
        elif self.encoder_type == 'vgg16':
            preprocessed_img = vgg16.preprocess_input(img)
        else:
            # Ensure values are between -1 and 1
            preprocessed_img = (img / np.max(img)) * 2 - 1
        return self.encoder(preprocessed_img)
