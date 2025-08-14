
import numpy as np
from mediapipe.framework.formats import landmark_pb2

def calculate_v_taper_ratio(landmarks: landmark_pb2.NormalizedLandmarkList, image_width: int, image_height: int) -> float:
    """
    Calculates the V-Taper (shoulder-to-waist) ratio from pose landmarks.

    Args:
        landmarks: The NormalizedLandmarkList from MediaPipe.
        image_width: The width of the image to un-normalize coordinates.
        image_height: The height of the image to un-normalize coordinates.

    Returns:
        The calculated V-Taper ratio, or 0.0 if landmarks are not sufficient.
    """
    if not landmarks:
        return 0.0

    # MediaPipe landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    # Get landmark coordinates
    left_shoulder_pt = landmarks.landmark[LEFT_SHOULDER]
    right_shoulder_pt = landmarks.landmark[RIGHT_SHOULDER]
    left_hip_pt = landmarks.landmark[LEFT_HIP]
    right_hip_pt = landmarks.landmark[RIGHT_HIP]

    # Check for landmark visibility
    if not (left_shoulder_pt.visibility > 0.5 and 
            right_shoulder_pt.visibility > 0.5 and 
            left_hip_pt.visibility > 0.5 and 
            right_hip_pt.visibility > 0.5):
        return 0.0 # Not enough data to calculate

    # Un-normalize coordinates to get pixel values
    ls_x = left_shoulder_pt.x * image_width
    rs_x = right_shoulder_pt.x * image_width
    lh_x = left_hip_pt.x * image_width
    rh_x = right_hip_pt.x * image_width

    # Calculate pixel widths
    shoulder_width = abs(ls_x - rs_x)
    hip_width = abs(lh_x - rh_x)

    if hip_width == 0:
        return 0.0 # Avoid division by zero

    # Calculate V-Taper ratio
    v_taper_ratio = shoulder_width / hip_width

    return v_taper_ratio

def get_v_taper_score(v_taper_ratio: float) -> int:
    """
    Converts a V-Taper ratio into a score out of 100.

    The scoring is based on proximity to the "golden ratio" of 1.618,
    which is often considered an aesthetic ideal.

    Args:
        v_taper_ratio: The calculated shoulder-to-waist ratio.

    Returns:
        A score from 0 to 100.
    """
    if v_taper_ratio == 0.0:
        return 0

    # Define the ideal and a baseline ratio
    ideal_ratio = 1.618
    baseline_ratio = 1.0  # Shoulders and waist are the same width

    # Calculate score based on a linear scale up to the ideal
    score = 100 * (v_taper_ratio - baseline_ratio) / (ideal_ratio - baseline_ratio)

    # Clamp the score to a maximum of 100, but allow for a minimum of 0
    score = min(max(score, 0), 100)

    return int(score)

# --- PLACEHOLDER FUNCTIONS for the Anatomist Module ---

def analyze_muscularity(image: np.ndarray) -> dict:
    """
    [PLACEHOLDER] Analyzes muscle fullness from a segmented image.
    This function will be replaced by a real U-Net model analysis.
    """
    # Dummy scores for demonstration
    return {
        "Pectorals": 85,
        "Deltoids": 92,
        "Abdominals": 88,
        "Overall Fullness": 88
    }

def analyze_conditioning(image: np.ndarray) -> dict:
    """
    [PLACEHOLDER] Analyzes muscle conditioning (separation, definition).
    This function will be replaced by a real model analysis.
    """
    # Dummy scores for demonstration
    return {
        "Separation": 82,
        "Abdominal Definition": 90,
        "Overall Conditioning": 86
    }

# --- HOLISTIC SCORING ---

def calculate_total_package_score(v_taper_score: int, muscularity_score: int, conditioning_score: int) -> int:
    """
    Calculates a holistic "Total Package Score" that penalizes imbalances.

    Args:
        v_taper_score: The score for symmetry/V-Taper.
        muscularity_score: The overall score for muscle fullness.
        conditioning_score: The overall score for conditioning.

    Returns:
        The final Total Package Score.
    """
    scores = np.array([v_taper_score, muscularity_score, conditioning_score])
    
    # Define weights for each category (can be tuned per division)
    # For Men's Physique: Symmetry > Conditioning > Muscularity
    weights = np.array([0.4, 0.3, 0.3])
    
    # Calculate the weighted average
    weighted_average = np.average(scores, weights=weights)
    
    # Calculate a penalty based on the standard deviation of the scores
    # A higher std dev means more imbalance, resulting in a larger penalty.
    std_dev = np.std(scores)
    penalty_factor = 1 - (std_dev / 100) # The 100 is a scaling factor, can be tuned
    
    # Apply the penalty
    final_score = weighted_average * penalty_factor
    
    return int(final_score)
