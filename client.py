from argparse import ArgumentParser
from facial_recognition_utils import recognise_face_in_image, recognise_face_in_video_stream


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--ip-address', type=str, default='127.0.0.1', dest='ip', help='Server ip address')
    parser.add_argument('-p', '--port', type=int, default=8000, dest='port', help='Server port')
    parser.add_argument('-f', '--face', type=str, dest='img_path', help='Image file path')
    parser.add_argument('-s', '--show-image', action='store_false',
                        dest='show', help='Show visualisation of facial recognition on image/video')
    return parser


if __name__ == '__main__':
    arg_parser = build_parser()
    args = arg_parser.parse_args()
    server_url = 'http://' + args.ip + ':' + str(args.port) + '/recognise'
    visualise = args.show
    if args.img_path is not None:
        recognise_face_in_image(args.img_path, url=server_url, visualise_result=visualise)
    else:
        recognise_face_in_video_stream(url=server_url, visualise_result=visualise)
