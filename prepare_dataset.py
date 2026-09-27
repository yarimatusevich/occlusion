from pathlib import Path

import cv2

from ml.face_crop import crop_face_from_frame

# resolve paths from this file's location, not the current working directory
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "ml" / "raw_data"
DATA_DIR = ROOT / "ml" / "data"
FRAME_STRIDE = 10  # keep every Nth frame


def convert_video_to_cropped_frames(video_path: Path, out_dir: Path, stride: int = FRAME_STRIDE):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = f"{num_frames / fps:.1f}s" if fps > 0 else "unknown"
    print(f"{video_path.name}: FPS={fps:.1f}, frames={num_frames}, duration={duration}")

    saved = skipped = 0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % stride == 0:
                face_crop = crop_face_from_frame(frame)
                if face_crop is None:
                    skipped += 1
                else:
                    # prefix with the video name so multiple videos per class don't overwrite each other
                    out_path = out_dir / f"{video_path.stem}_frame{idx:06d}.jpg"
                    cv2.imwrite(str(out_path), face_crop)
                    saved += 1
            idx += 1
    finally:
        cap.release()

    print(f"{video_path.name}: saved {saved} crops, no face in {skipped} sampled frames")
    return saved, skipped


if __name__ == "__main__":
    jobs = [
        (RAW_DIR / "looking.mov", DATA_DIR / "looking"),
        (RAW_DIR / "not_looking.mov", DATA_DIR / "not_looking"),
    ]
    for video_path, out_dir in jobs:
        convert_video_to_cropped_frames(video_path, out_dir)