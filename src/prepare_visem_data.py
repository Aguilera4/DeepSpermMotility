import os
import cv2
import pandas as pd
from tqdm import tqdm

VIDEO_DIR = "videos/"
TRACKING_DIR = "tracking/"
OUTPUT_DIR = "dataset/"
FRAME_RATE = 50  # frames per second
IMAGE_SIZE = (640, 480)

def ensure_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

def extract_and_label(video_path, tracking_path, split="train"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps / FRAME_RATE)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    df = pd.read_csv(tracking_path)

    frame_idx = 0
    count = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Processing {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_idx:04d}"
            image_path = os.path.join(OUTPUT_DIR, "images", split, frame_name + ".jpg")
            label_path = os.path.join(OUTPUT_DIR, "labels", split, frame_name + ".txt")
            cv2.imwrite(image_path, frame)

            # Get tracking annotations for current frame
            annots = df[df['frame'] == count]
            with open(label_path, "w") as f:
                for _, row in annots.iterrows():
                    # Assume format: frame, id, x, y, w, h
                    x_center = (row["x"] + row["w"] / 2) / width
                    y_center = (row["y"] + row["h"] / 2) / height
                    w_norm = row["w"] / width
                    h_norm = row["h"] / height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            frame_idx += 1
        count += 1
        pbar.update(1)
    cap.release()
    pbar.close()

# Example usage
ensure_dirs()
extract_and_label("videos/sample1.mp4", "tracking/sample1.csv", split="train")