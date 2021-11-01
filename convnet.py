import tensorflow as tf
import tensorflow.keras.layers as layers


def conv_block(n_filters, inputs):
    x = layers.Conv2D(n_filters, 3, padding='same', dilation_rate=1)(inputs)
    x = layers.Conv2D(n_filters, 3, padding='same', dilation_rate=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    return x


def build_model(n_blocks=5,
                input_size=(134, 134, 3),
                uniform_filters=False,
                predict_patches=True,
                n_patches=10,
                patch_size=(15, 15)):
    input_ = tf.keras.Input(input_size)
    x = None
    scale = 1
    for idx in range(n_blocks):
        if idx == 0:
            x = conv_block(32*scale, input_)
        else:
            x = conv_block(32*scale, x)
        if idx % 2 == 1 and not uniform_filters:
            scale += 1
    encoder = tf.keras.Model(input_, x, name='convnet_face_encoder')
    if predict_patches:
        overlap = (input_size[0] % patch_size[0], input_size[1] % patch_size[1])
        pool = layers.AvgPool2D((overlap[0]+input_size[0]//patch_size[0], overlap[1]+input_size[1]//patch_size[1]),
                                (input_size[0]//patch_size[0], input_size[1]//patch_size[1]))(x)
        output = layers.Conv2D(3*n_patches, 1, activation='sigmoid', padding='same')(pool)
    else:
        reconstruction = layers.Conv2D(3, 1, activation='sigmoid', padding='same')(x)
        output = layers.Add()([input_, reconstruction])
    full_model = tf.keras.Model(input_, output, name='convnet_face_reconstructor')
    return {'full_model': full_model, 'encoder': encoder}


if __name__ == '__main__':
    convnet = build_model()['full_model']
    print(convnet.summary())
