import numpy as np
import tensorflow.keras.applications.xception as xception
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.applications.vgg16 as vgg16

encoder_types = ['xception', 'resnet50', 'vgg16']


class FaceEncoder:
    def __init__(self, encoder_type='xception'):
        assert encoder_type in encoder_types, 'Invalid encoder type'
        self.encoder_type = encoder_type
        if self.encoder_type == 'xception':
            self.encoder = xception.Xception(include_top=False, pooling='avg')
        elif self.encoder_type == 'resnet50':
            self.encoder = resnet50.ResNet50(include_top=False, pooling='avg')
        elif self.encoder_type == 'vgg16':
            self.encoder = vgg16.VGG16(include_top=False, pooling='avg')

    def encode(self, img):
        if len(img.shape) == 3:
            img = np.array([img])
        if self.encoder_type == 'xception':
            encoded_img = self.encoder(xception.preprocess_input(img))
        elif self.encoder_type == 'resnet50':
            encoded_img = self.encoder(resnet50.preprocess_input(img))
        elif self.encoder_type == 'vgg16':
            encoded_img = self.encoder(vgg16.preprocess_input(img))
        return encoded_img
