import tensorflow as tf
import numpy as np
from importlib import import_module
from matplotlib import pyplot
from image_transform_utils import mask_random_pixels, mask_random_areas
from face_detector import FaceDetector

MODEL_TYPE = 'convnet'
IMG_PATH = 'faces'
TEST_IMG_PATH = 'test_faces'
IMAGE_SIZE = (128, 128)
USE_FACE_DETECTOR = False
PREDICT_PATCHES_ONLY = True
TRAINING_EPOCHS = 10
BATCH_SIZE = 48
VALIDATION_SPLIT = 0.2
TRAINING_STEPS = 2000
VALIDATION_STEPS = int(TRAINING_STEPS * VALIDATION_SPLIT/(1-VALIDATION_SPLIT))
SEED = 123
MASK_AREAS = True
MASK_RATIO = 0.4
MASK_SIZE = (15, 15)
LOSS = tf.keras.losses.MeanAbsoluteError()
np.random.seed(SEED)

# Dynamically import build_model function based on MODEL_TYPE
module = import_module(MODEL_TYPE)
build_model = getattr(module, 'build_model')


def image_data_generator(iterator, mask_areas=MASK_AREAS, patches_only=PREDICT_PATCHES_ONLY, test_mode=False):
    face_detector = FaceDetector(face_size=IMAGE_SIZE)
    for imgs in iterator:
        if USE_FACE_DETECTOR:
            batch_faces = []
            for img in imgs:
                faces = face_detector.detect_faces(img)
                if len(faces) > 0:
                    for face in faces:
                        batch_faces.append(np.array(face['image']))
            batch_faces = np.array(batch_faces)
        else:
            batch_faces = imgs
        if mask_areas:
            mask, patches, mask_indices = mask_random_areas(batch_faces, mask_size=MASK_SIZE)
        else:
            mask = mask_random_pixels(batch_faces, mask_ratio=MASK_RATIO)
        masked_faces = batch_faces * mask
        if mask_areas and patches_only:
            if test_mode:
                yield masked_faces, patches, mask_indices, batch_faces
            else:
                yield masked_faces, patches
        else:
            yield masked_faces, batch_faces


def build_image_data_generators():

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
                                                              vertical_flip=True,
                                                              horizontal_flip=True,
                                                              rotation_range=30,
                                                              validation_split=.2)

    train = img_gen.flow_from_directory(IMG_PATH,
                                        class_mode=None,
                                        target_size=IMAGE_SIZE,
                                        batch_size=BATCH_SIZE,
                                        seed=SEED,
                                        subset='training')

    val = img_gen.flow_from_directory(IMG_PATH,
                                      class_mode=None,
                                      target_size=IMAGE_SIZE,
                                      batch_size=BATCH_SIZE,
                                      seed=SEED,
                                      subset='validation')

    test = img_gen.flow_from_directory(TEST_IMG_PATH,
                                       class_mode=None,
                                       target_size=IMAGE_SIZE,
                                       batch_size=BATCH_SIZE,
                                       seed=SEED,
                                       subset='validation')

    return image_data_generator(train), image_data_generator(val), image_data_generator(test, test_mode=True)


def rebuild_single_image_from_predictions(masked_img, patches, ind):
    img = np.copy(masked_img)
    if len(img.shape) == 4:
        img = np.squeeze(img, 0)
    if len(patches.shape) == 4:
        patches = np.squeeze(patches, 0)
    for patch_idx, start in zip(range(0, 30, 3), ind):
        patch = patches[:, :, patch_idx:patch_idx+3]
        img[start[0]:start[0]+MASK_SIZE[0], start[1]:start[1]+MASK_SIZE[1]] = patch
    return img


if __name__ == '__main__':
    # Build the image data generators for training and validation
    train_generator, validation_generator, test_generator = build_image_data_generators()

    # Build the model and split out the encoder component
    face_model = build_model(input_size=[*IMAGE_SIZE, 3], predict_patches=PREDICT_PATCHES_ONLY, patch_size=MASK_SIZE)
    face_reconstructor = face_model['full_model']
    face_encoder = face_model['encoder']
    face_reconstructor.compile('adam', LOSS)

    # Train the full model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{MODEL_TYPE}_model.h5', save_best_only=True)
    stop_early = tf.keras.callbacks.EarlyStopping(patience=5)
    history = face_reconstructor.fit(train_generator,
                                     validation_data=validation_generator,
                                     epochs=TRAINING_EPOCHS,
                                     steps_per_epoch=TRAINING_STEPS,
                                     validation_steps=VALIDATION_STEPS,
                                     callbacks=[checkpoint])

    # Save encoder component
    face_encoder.save(f'{MODEL_TYPE}_encoder.h5')

    # Plot losses
    pyplot.figure()
    pyplot.title('Training Losses')
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.legend(['loss', 'val_loss'])

    if PREDICT_PATCHES_ONLY:
        # Run inference on an input sample
        test_input, ground_truth, indices, original = next(test_generator)
        test_input, ground_truth, indices, original = test_input[0], ground_truth[0], indices[0], original[0]
        test_input = np.array([test_input])
        test_output = face_reconstructor.predict(test_input)
        # Rebuild the image using predicted patches
        model_reconstruction = rebuild_single_image_from_predictions(test_input, test_output, indices)
        # Visualise the model in action
        pyplot.figure()
        pyplot.subplot(131)
        pyplot.title('Input')
        pyplot.imshow(test_input[0])
        pyplot.subplot(132)
        pyplot.title('Ground Truth')
        pyplot.imshow(original)
        pyplot.subplot(133)
        pyplot.title('Reconstruction')
        pyplot.imshow(model_reconstruction)
    else:
        # Run inference on an input sample
        test_input, ground_truth = next(test_generator)
        test_input, ground_truth = test_input[0], ground_truth[0]
        test_input = np.array([test_input])
        test_output = face_reconstructor.predict(test_input)
        # Visualise the model in action
        pyplot.figure()
        pyplot.subplot(131)
        pyplot.title('Input')
        pyplot.imshow(test_input[0])
        pyplot.subplot(132)
        pyplot.title('Ground Truth')
        pyplot.imshow(ground_truth)
        pyplot.subplot(133)
        pyplot.title('Reconstruction')
        pyplot.imshow(test_output[0])

    # Show plots
    pyplot.show()
