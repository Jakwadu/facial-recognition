from argparse import ArgumentParser
from facial_recogniser import FacialRecogniser
from facial_recognition_utils import recognise_face_in_image, recognise_face_in_video_stream


DEFAULT_REFERENCE_FACE_PATH = 'references'
SIMILARITY_THRESHOLD = 0.6


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('-i', '--image', type=str,
                        dest='image', help='Path to file if doing facial recognition on an image', required=False)

    parser.add_argument('-r', '--reference-faces', type=str, dest='reference_faces',  required=False,
                        default=DEFAULT_REFERENCE_FACE_PATH,
                        help='Directory containing examples of the person of interest')

    parser.add_argument('-m', '--mtcnn', dest='mtcnn', action='store_true', help='Use MTCNN for face detection')

    parser.add_argument('-t', '--threshold', type=float, dest='threshold', required=False, default=SIMILARITY_THRESHOLD,
                        help='Euclidean distance threshold for image similarity')

    parser.add_argument('-n', '--no-alignment', dest='no_alignment', action='store_true',
                        help='Disable facial alignment after face detection')

    return parser


if __name__ == '__main__':
    arg_parser = build_parser()
    args = arg_parser.parse_args()
    recogniser = FacialRecogniser(args.reference_faces,
                                  threshold=args.threshold,
                                  align=not args.no_alignment,
                                  mtcnn=args.mtcnn)
    image_path = args.image
    if not image_path:
        print(f'***** Running facial recognition on video stream')
        recognise_face_in_video_stream(recogniser)
    else:
        print(f'***** Running facial recognition on {image_path}')
        recognise_face_in_image(image_path, recogniser)
