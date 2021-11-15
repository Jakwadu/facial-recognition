from argparse import ArgumentParser
from facial_recogniser import FacialRecogniser
from facial_recognition_utils import recognise_face_in_image, recognise_face_in_video_stream


DEFAULT_REFERENCE_FACE_PATH = 'references'
ENCODER_TYPE = 'vgg-face'
SIMILARITY_THRESHOLD = 0.4


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('-i', '--image', type=str,
                        dest='image', help='Path to file if doing facial recognition on an image',
                        metavar='IMAGE', required=False)

    parser.add_argument('-r', '--reference-faces', type=str,
                        dest='reference_faces', help='Directory containing examples of the person of interest',
                        metavar='REFERENCE_FACES', required=False, default=DEFAULT_REFERENCE_FACE_PATH)

    return parser


if __name__ == '__main__':
    arg_parser = build_parser()
    args = arg_parser.parse_args()
    reference_faces = args.reference_faces
    recogniser = FacialRecogniser(ENCODER_TYPE, reference_faces, threshold=SIMILARITY_THRESHOLD)
    image_path = args.image
    if not image_path:
        print(f'***** Running facial recognition on video stream')
        recognise_face_in_video_stream(recogniser)
    else:
        print(f'***** Running facial recognition on {image_path}')
        recognise_face_in_image(image_path, recogniser)
