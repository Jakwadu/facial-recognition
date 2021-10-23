import tensorflow as tf
import tensorflow.keras.layers as layers


def conv_block(n_filters, inputs):
    x = layers.Conv2D(n_filters, 3, padding='same')(inputs)
    x = layers.Conv2D(n_filters, 3, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    return x


def encoder_block(n_filters, inputs):
    skip = conv_block(n_filters, inputs)
    out = layers.MaxPool2D((2,2))(skip)
    return out, skip


def decoder_block(n_filters, inputs, skip):
    x = layers.Conv2DTranspose(n_filters, (2,2), strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip])
    x = conv_block(n_filters, x)
    return x


def build_model(n_blocks=2, input_size=(300, 300, 3), predict_patches=True, patch_size=(3, 3)):
    input_ = tf.keras.Input(input_size)
    x = None
    skips = []
    for i in range(n_blocks*2):
        idx = i % n_blocks
        if i < n_blocks:
            if idx == 0:
                x, s = encoder_block(64, input_)
            else:
                x, s = encoder_block(64*(1+idx), x)
                if idx == n_blocks - 1:
                    x = conv_block(64*(2+idx), x)
            skips.append(s)
        else:
            x = decoder_block(64*(n_blocks-idx), x, skips[-1-idx])
    reconstruction = layers.Conv2D(3, 1, activation='sigmoid', padding='same')(x)
    output = layers.Add()([input_, reconstruction])
    encoder = tf.keras.Model(input_, x, name='unet_face_encoder')
    full_model = tf.keras.Model(input_, output, name='unet_face_reconstructor')
    return {'full_model': full_model, 'encoder': encoder}


if __name__ == '__main__':
    u_net = build_model()['encoder']
    print('### Built UNet with default parameters')
    print(u_net.summary())
