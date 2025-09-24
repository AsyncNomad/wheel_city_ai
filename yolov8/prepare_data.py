import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration (All relative paths) ---
# Dataset base path (current folder)
base_path = Path(".")
original_image_path = base_path / "images"
xml_file_path = base_path / "wm_annotations.xml"

# Path for the new dataset
output_path = base_path / "datasets"

# Class mapping definition
# Original label -> New class ID (curb: 0, ramp: 1)
class_mapping = { "step": 0, "stair": 0, "ramp": 1 }

def convert_to_yolo(size, box):
    """Converts Pascal VOC(xtl, ytl, xbr, ybr) to YOLO(center_x, center_y, width, height) format"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw; w = w * dw; y = y * dh; h = h * dh
    return (x, y, w, h)

# --- 1. Parse XML and create an image list ---
print(f"1. Parsing '{xml_file_path}'...")
tree = ET.parse(xml_file_path)
root = tree.getroot()

images_with_ramp = []
images_with_curb_only = []
image_annotations = {}

for image_elem in root.findall("image"):
    img_name = Path(image_elem.get("name")).name
    img_width = int(image_elem.get("width"))
    img_height = int(image_elem.get("height"))
    
    full_img_path = original_image_path / img_name
    
    if not full_img_path.exists():
        print(f"Warning: Image not found at {full_img_path}. Skipping.")
        continue

    contains_ramp, contains_curb = False, False
    boxes = []

    for box_elem in image_elem.findall("box"):
        label = box_elem.get("label")
        if label in class_mapping:
            if label == "ramp": contains_ramp = True
            else: contains_curb = True
            
            xtl, ytl = float(box_elem.get("xtl")), float(box_elem.get("ytl"))
            xbr, ybr = float(box_elem.get("xbr")), float(box_elem.get("ybr"))
            
            yolo_box = convert_to_yolo((img_width, img_height), (xtl, xbr, ytl, ybr))
            boxes.append(f"{class_mapping[label]} {' '.join(map(str, yolo_box))}")

    if boxes:
        image_annotations[full_img_path] = boxes
        if contains_ramp: images_with_ramp.append(full_img_path)
        elif contains_curb: images_with_curb_only.append(full_img_path)

print(f" - Images containing 'ramp': {len(images_with_ramp)}")
print(f" - Images containing 'curb' only: {len(images_with_curb_only)}")

# --- 2. Perform data undersampling ---
print("\n2. Performing data undersampling...")
num_ramps = len(images_with_ramp)
if len(images_with_curb_only) > num_ramps:
    sampled_curb_images = random.sample(images_with_curb_only, num_ramps)
    print(f" - Undersampling 'curb' images to {num_ramps}.")
else:
    sampled_curb_images = images_with_curb_only

final_image_list = images_with_ramp + sampled_curb_images
random.shuffle(final_image_list)
print(f" - Final training dataset size: {len(final_image_list)} images")

# --- 3. Split data and generate files ---
print("\n3. Creating YOLO dataset...")
train_files, val_files = train_test_split(final_image_list, test_size=0.2, random_state=42)

if output_path.exists(): shutil.rmtree(output_path)

def create_dataset_files(file_list, subset):
    img_dir = output_path / subset / "images"
    lbl_dir = output_path / subset / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in file_list:
        shutil.copy(img_path, img_dir)
        label_filename = img_path.stem + ".txt"
        with open(lbl_dir / label_filename, 'w') as f:
            f.write("\n".join(image_annotations[img_path]))

create_dataset_files(train_files, "train")
create_dataset_files(val_files, "val")
print(f"\n✅ Success! YOLO dataset created in '{output_path}' folder.")