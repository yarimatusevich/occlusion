import cv2
import torch
import cv2 as cv
from model import inference_transform, CNN
from preprocess_data import crop_face_from_frame

LOOKING_THRESHOLD = 0.5

def detect_gaze(model, frame):
    # cropping face from video frame and converting to tensors
    face_crop = crop_face_from_frame(frame)

    if face_crop is None:
        # if face not found return not_looking class
        return 1

    face_crop_transformed = inference_transform(face_crop)

    # take class with the greatest probability as prediction
    scores = model(face_crop_transformed.unsqueeze(0))
    probs = torch.softmax(scores, 1)
    looking_confidence = probs[0][0]

    # thresholding model has to be at least 75% user is looking to output looking class
    prediction = 0 if looking_confidence > LOOKING_THRESHOLD else 1

    return prediction

def start_live_webcam_feed(model, max_length_frames=None, save_labeled_frames=False):
    webcam_capture = cv.VideoCapture(0)

    if not webcam_capture.isOpened():
        return

    # keeping track of frames
    frame_num = 0
    prediction_str = 'Not Looking'

    while True:
        if max_length_frames and frame_num >= max_length_frames:
            break

        ret, frame = webcam_capture.read()

        if not ret:
            print('Failed to capture frame')
            continue

        if frame_num % 10 == 0:
            # gaze detection (looking, not looking
            prediction = detect_gaze(model, frame)
            prediction_str = 'Not Looking' if prediction == 1 else 'Looking'

        # displaying feed with class prediction
        cv.putText(
            frame, prediction_str, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 0), 2
        )

        cv.imshow('Webcam Feed', frame)

        # saves frame with label from model if true
        if save_labeled_frames:
            cv.imwrite(f'labeled_data/frame_{frame_num}.png', frame)

        # press 'q' to stop feed
        if cv.waitKey(1) == ord('q'):
            break

        frame_num += 1

    webcam_capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    model = CNN()
    state_dict = torch.load('model.pth', weights_only=True)

    # loading model and setting to evaluation mode
    model.load_state_dict(state_dict)
    model.eval()

    # detecting gaze through live webcam feed
    start_live_webcam_feed(model, 500, save_labeled_frames=True)