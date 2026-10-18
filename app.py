import cv2 as cv
import gradio as gr

from ml import inference

# using free hosting tier with access to cpu only
model = inference.load_model(map_location='cpu')


def predict(frame):
    """Return (class confidences, face crop the model saw) for an RGB webcam frame."""
    if frame is None:
        return None, None

    # gradio gives RGB frames, the face detector expects BGR like opencv's VideoCapture
    result = inference.predict(model, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
    if result is None:
        return {'No face detected': 1.0}, None

    probs, face_crop_rgb = result
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
