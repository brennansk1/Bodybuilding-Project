

import os
import argparse
import cv2
import numpy as np
from PIL import Image
import subprocess
import io

# To make this script runnable from the root directory, we add the project path.
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# We only need normalize_lighting from our module now
from poseperfect_ai.preprocessing.image_preprocessor import normalize_lighting

def process_images(input_dir: str, output_dir: str):
    """
    Processes all images in a directory by removing the background and normalizing
    lighting, then saves them to an output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    image_files = [f for f in os.path.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process.")

    for filename in image_files:
        print(f"\n--- Processing {filename} ---")
        # Define paths
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        # Create a temporary path for the background-removed image
        temp_bg_removed_path = os.path.join(output_dir, f"temp_{filename}")

        try:
            # Step 1: Remove background using rembg CLI via subprocess
            print(f"  [1/4] Removing background (using CLI method)... ")
            command = ["rembg", "i", input_path, temp_bg_removed_path]
            # Using capture_output to hide rembg's own console output for a cleaner log
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            # Step 2: Read the background-removed image for further processing
            print(f"  [2/4] Reading temporary file...")
            with open(temp_bg_removed_path, 'rb') as f:
                image_bytes_no_bg = f.read()
            
            no_bg_image_pil = Image.open(io.BytesIO(image_bytes_no_bg))
            if no_bg_image_pil.mode == 'RGBA':
                no_bg_image_pil = no_bg_image_pil.convert('RGB')
            image_np = np.array(no_bg_image_pil)

            # Step 3: Lighting Normalization
            print("  [3/4] Normalizing lighting...")
            normalized_image = normalize_lighting(image_np)
            
            output_image_bgr = cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR)

            # Step 4: Save the final processed image
            print(f"  [4/4] Saving final image to: {output_path}")
            success = cv2.imwrite(output_path, output_image_bgr)
            if not success:
                print(f"  [ERROR] Failed to save image to {output_path}.")
            else:
                print(f"  -> Successfully saved {filename}")

        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Background removal failed for {filename}.")
            print(f"  - STDERR: {e.stderr}")
        except Exception as e:
            print(f"  [UNEXPECTED ERROR] Failed to process {filename}: {e}", flush=True)
        finally:
            # Clean up the temporary file if it exists
            if os.path.exists(temp_bg_removed_path):
                os.remove(temp_bg_removed_path)

    print(f"\n--- All processing complete. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for dataset creation.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the raw input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where processed images will be saved.")

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir)
