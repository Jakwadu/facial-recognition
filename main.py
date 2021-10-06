import cv2
import matplotlib.patches as patches
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from facial_recogniser import FacialRecogniser


DEFAULT_REFERENCE_FACE_PATH = 'target_faces/test'
DEFAULT_IMAGE = 'target_faces/sources/37w2_5d2Y4r_l.jpg'
ENCODER_TYPE = 'xception'
SIMILARITY_THRESHOLD = 0.6


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('-i', '--image', type=str,
                        dest='image', help='Path to file if doing facial recognition on an image',
                        metavar='IMAGE', required=False)

    parser.add_argument('-r', '--reference-faces', type=str,
                        dest='reference_faces', help='Directory containing examples of the target face',
                        metavar='REFERENCE_FACES', required=False, default=DEFAULT_REFERENCE_FACE_PATH)

    return parser


def recognise_face_in_video_stream(facial_recogniser):
    camera = cv2.VideoCapture(0)
    camera_ok = camera.isOpened()
    if not camera_ok:
        print('Failed to open camera')
        input('Press RETURN to exit')
    else:
        while camera_ok:
            camera_ok, frame = camera.read()
            result = facial_recogniser.recognise_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(result) > 0:
                for r in result:
                    x0, y0, x1, y1 = r['face']['co-ordinates']
                    recognised = r['recognised']
                    text = 'Haaland' if recognised else 'Unknown'
                    colour = (0, 255, 0) if recognised else (0, 0, 255)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
                    cv2.putText(frame, text, (x1 + 10, y1), 0, 0.3, colour)
            cv2.imshow('Facial Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.release()


def recognise_face_in_image(img_path, facial_recogniser):
    img = plt.imread(img_path)
    result = facial_recogniser.recognise_faces(img)
    fig, ax = plt.subplots()
    ax.imshow(img)
    if len(result) > 0:
        for r in result:
            x0, y0, x1, y1 = r['face']['co-ordinates']
            recognised = r['recognised']
            colour = 'g' if recognised else 'r'
            rectangle = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor=colour, facecolor='none')
            ax.add_patch(rectangle)
            text = 'Haaland' if recognised else 'Unknown'
            ax.set_title(text)
    plt.show()


if __name__ == '__main__':
    arg_parser = build_parser()
    args = arg_parser.parse_args()
    reference_faces = args.reference_faces
    recogniser = FacialRecogniser(reference_faces, ENCODER_TYPE, threshold=SIMILARITY_THRESHOLD)
    image_path = args.image
    if not image_path:
        print(f'***** Running facial recognition on video stream')
        recognise_face_in_video_stream(recogniser)
    else:
        print(f'***** Running facial recognition on {image_path}')
        recognise_face_in_image(image_path, recogniser)
