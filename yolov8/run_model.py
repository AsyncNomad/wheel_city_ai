from pathlib import Path
from ultralytics import YOLO

# --- Configuration (Relative Paths) ---
# Path to the trained YOLOv8 model weights
MODEL_PATH = "./runs/detect/wheel_city_ai_v1/weights/best.pt"
# Directory containing test images
TEST_IMAGES_DIR = "./test_images"
# Directory to save visualized images
OUTPUT_VISUALIZATIONS_DIR = Path("./test_results/visualizations")

# --- 1. Load the trained model ---
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load model. Please ensure the model file exists at '{MODEL_PATH}' and training is complete.")
    print(f"Details: {e}")
    exit(1)

# Ensure output directory exists
OUTPUT_VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_and_visualize(image_path: Path):
    """
    Analyzes a single image, prints the accessibility result,
    and saves the visualized image with bounding boxes.
    """
    print(f"\n--- Analyzing: {image_path.name} ---")

    try:
        # Perform inference
        results = model(image_path, verbose=False) # verbose=False to suppress detailed output
    except Exception as e:
        print(f"ERROR: An error occurred during inference for {image_path.name}: {e}")
        return

    has_curb = False
    has_ramp = False
    
    # Process the results
    if results:
        r = results[0] # Get the first result object
        
        # Check for any detections
        if r.boxes:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name == 'curb':
                    has_curb = True
                elif class_name == 'ramp':
                    has_ramp = True
        
        # Save visualization (image with bounding boxes)
        visualized_image_path = OUTPUT_VISUALIZATIONS_DIR / f"annotated_{image_path.name}"
        r.save(filename=visualized_image_path)
        print(f"Visualized image saved to: {visualized_image_path}")
    else:
        print(f"WARNING: No objects detected or an issue occurred for {image_path.name}.")

    # Apply accessibility logic: accessible = (~curb) || (curb && ramp)
    is_accessible = (not has_curb) or (has_curb and has_ramp)

    #print(f"  Curb/Stair Detected: {has_curb}")
    #print(f"  Ramp Detected: {has_ramp}")
    #print(f"  Accessibility: {'Accessible' if is_accessible else 'Not Accessible'}")

def main():
    """
    Main function to find all images in the test directory and analyze them.
    """
    # Get all image files from the test directory
    image_files = [f for f in Path(TEST_IMAGES_DIR).iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not image_files:
        print(f"No image files found in '{TEST_IMAGES_DIR}'. Please put your test images there.")
        return

    print(f"\nFound {len(image_files)} images to analyze.")
    for image_file in sorted(image_files): # Sort for consistent order
        analyze_and_visualize(image_file)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()