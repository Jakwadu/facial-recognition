import cv2
import matplotlib.patches as patches
from matplotlib import pyplot as plt


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
                    text = r['name'] if recognised else 'Unknown'
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
            text = r['name'] if recognised else 'Unknown'
            ax.set_title(text)
    plt.show()
