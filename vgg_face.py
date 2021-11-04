import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


def build_encoder():
    model = tf.keras.Sequential()
    model.add(layers.ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Convolution2D(4096, (7, 7), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Convolution2D(4096, (1, 1), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Convolution2D(2622, (1, 1)))
    model.add(layers.Flatten())
    model.add(layers.Activation('softmax'))

    model.load_weights('vgg_face_weights.h5')
    encoder = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    encoder.trainable = False

    return encoder


if __name__ == '__main__':
    vgg_face_encoder = build_encoder()
    print(vgg_face_encoder.summary())
    input_array = np.random.random((1, 224, 224, 3))
    print('\nInput array shape:', input_array.shape)
    output_array = vgg_face_encoder(input_array)
    print('\nOutput array shape:', output_array.shape)
    print(np.max(output_array), '|', np.min(output_array))
