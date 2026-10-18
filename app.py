from pathlib import Path

import cv2 as cv
import gradio as gr
import torch
from PIL import Image

from ml.occlusion_model import OcclusionModel
from ml.data_transforms import transform_image_for_inference
from ml.face_crop import crop_face_from_frame

MODEL_PATH = Path(__file__).resolve().parent / "ml" / "models" / "model.pth"


def load_model():
    model = OcclusionModel()
    # using free hosting tier with access to cpu only
    state_dict = torch.load(MODEL_PATH, weights_only=True, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model


model = load_model()


def predict(frame):
    """Return (class confidences, face crop the model saw) for an RGB webcam frame."""
    if frame is None:
        return None, None

    # gradio gives RGB frames, the face detector expects BGR like opencv's VideoCapture
    face_crop = crop_face_from_frame(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
    if face_crop is None:
        return {'No face detected': 1.0}, None

    face_crop_rgb = cv.cvtColor(face_crop, cv.COLOR_BGR2RGB)
    face_crop_transformed = transform_image_for_inference(Image.fromarray(face_crop_rgb))

    with torch.no_grad():
        scores = model(face_crop_transformed.unsqueeze(0))
    probs = torch.softmax(scores, 1)[0]

    # class indices follow ImageFolder's alphabetical order: looking, not_looking
    return {'Looking': probs[0].item(), 'Not Looking': probs[1].item()}, face_crop_rgb


with gr.Blocks(title='Occlusion: Gaze Detection') as demo:
    gr.Markdown(
        '# Gaze Detection\n'
        'Allow webcam access, then look at and away from the screen. '
        'Faces are detected with YuNet, cropped, and classified by a small CNN.'
    )
    with gr.Row():
        webcam = gr.Image(sources=['webcam'], streaming=True, label='Webcam')
        with gr.Column():
            label = gr.Label(num_top_classes=2, label='Prediction')
            face = gr.Image(label='Face crop seen by the model', height=200)

    webcam.stream(predict, inputs=webcam, outputs=[label, face], stream_every=0.2)


if __name__ == '__main__':
    demo.launch()
