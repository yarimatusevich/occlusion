import cv2 as cv
import torch
from PIL import Image

from ml.face_crop import crop_face_from_frame
from ml.model import OcclusionModel
from ml.paths import MODEL_PATH
from ml.transforms import transform_image_for_inference


def load_model(map_location='cpu'):
    """Load the trained model in evaluation mode."""
    model = OcclusionModel()
    state_dict = torch.load(MODEL_PATH, weights_only=True, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict(model, frame):
    """Return (probs, face_crop_rgb) for a BGR frame, or None if no face was found.

    probs is a tensor of [looking, not_looking], following ImageFolder's alphabetical class order.
    """
    face_crop = crop_face_from_frame(frame)
    if face_crop is None:
        return None

    # opencv frames are BGR numpy arrays, transforms expect an RGB PIL image
    face_crop_rgb = cv.cvtColor(face_crop, cv.COLOR_BGR2RGB)
    face_crop_transformed = transform_image_for_inference(Image.fromarray(face_crop_rgb))

    with torch.no_grad():
        scores = model(face_crop_transformed.unsqueeze(0))
    probs = torch.softmax(scores, 1)[0]

    return probs, face_crop_rgb
