import numpy as np
from PIL import Image
from rembg import remove
import io
import cv2
import mediapipe as mp
from typing import Tuple

def remove_background(image_bytes: bytes) -> Image.Image:
    """
    Removes the background from an image.
    """
    output_bytes = remove(image_bytes)
    return Image.open(io.BytesIO(output_bytes))

def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE to normalize image lighting.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_channel = clahe.apply(l_channel)
    merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

def detect_and_draw_landmarks(image: np.ndarray) -> Tuple[np.ndarray, any]:
    """
    Detects pose landmarks and draws them on the image.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    results = pose.process(image)
    
    annotated_image = image.copy()
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    return annotated_image, results.pose_landmarks

def preprocess_for_static_analysis(image_bytes: bytes) -> Tuple[np.ndarray, any]:
    """
    The main preprocessing pipeline for the Streamlit app's static analysis.
    """
    # Step 1: Background Removal
    no_bg_image_pil = remove_background(image_bytes)
    if no_bg_image_pil.mode == 'RGBA':
        no_bg_image_pil = no_bg_image_pil.convert('RGB')
    image_np = np.array(no_bg_image_pil)

    # Step 2: Lighting Normalization
    normalized_image = normalize_lighting(image_np)

    # Step 3: Landmark Detection
    annotated_image, landmarks = detect_and_draw_landmarks(normalized_image)

    return annotated_image, landmarks