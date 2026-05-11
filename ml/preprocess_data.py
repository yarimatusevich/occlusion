import cv2
import os

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face_from_frame(frame):
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        grey_image,
        1.1, # how much image is scaled down at each of detection
        5, # num of overlapping detection needed for region to be counted as a real object
        minSize=(40, 40) # min size of detection window
    )

    # model found no face
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    # adding padding to ensure entire face fits in bounding box
    pad = int(0.2 * h)

    # bounding box coordinates
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    face_crop = frame[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (128, 128))

    return face_crop

def convert_video_to_cropped_frames(video_filepath, is_looking=True):
    video_capture = cv2.VideoCapture(video_filepath)

    if not video_capture:
        print('ERROR: Failed to open video capture')
        exit()


    fps = video_capture.get(cv2.CAP_PROP_FPS)
    num_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'FPS: {fps}, Total frames: {num_frames}, Duration: {num_frames / fps:.1f}s')

    count = 0
    while True:
        success, frame = video_capture.read()

        if not success:
            break

        if count % 10 == 0:
            print(f'Cropping frame: {count}')
            face_crop = crop_face_from_frame(frame)

            # splitting into classes in data folder
            if face_crop is not None:
                if is_looking:
                    cv2.imwrite(f'data/looking/frame_{count}.jpg', face_crop)
                else:
                    cv2.imwrite(f'data/not_looking/frame_{count}.jpg', face_crop)

        count += 1

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.makedirs('data/train/looking', exist_ok=True)
    os.makedirs('data/train/not_looking', exist_ok=True)

    convert_video_to_cropped_frames('raw_data/looking.mov')
    convert_video_to_cropped_frames(video_filepath='raw_data/not_looking.mov', is_looking=False)