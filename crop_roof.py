import cv2
from pathlib import Path
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/sdowell1/1_finalProject/yolov5/runs/train/exp15/weights/best.pt')

input_dir = Path('/home/sdowell1/1_finalProject/yolov5/more_data/more_data/images')  
output_dir = Path('cropped_roofs')  
output_dir.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = 0.0  

for image_path in input_dir.glob('*.jpg'):  
    print(f"Processing: {image_path}")
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        print(f"Could not read image: {image_path}")
        raise Exception("Image cannot be read")

    results = model(str(image_path))  
    detections = results.xyxy[0]
    print(f"Detections for {image_path}: {detections}")
    
    if len(detections) == 0:
        print(f"No objects detected in {image_path}")
        continue
        
    for i, det in enumerate(detections):
        x_min, y_min, x_max, y_max, confidence, cls = map(int, det[:6])
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Skipping object {i} with low confidence: {confidence}")
            raise Exception("Prediction has too low of a confidence threshold")
        print(f"Bounding box {i}: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, confidence={confidence}")
        if x_min < 0 or y_min < 0 or x_max > original_image.shape[1] or y_max > original_image.shape[0]:
            print(f"Invalid bounding box for object {i}")
            raise Exception("Invalid bounding boxes")

        cropped_image = original_image[y_min:y_max, x_min:x_max]
        if cropped_image.size == 0:
            print(f"Empty cropped image for object {i}")
            raise Exception("Empty image")

        output_file = output_dir / f"{image_path.stem}_object_{i}.jpg"
        cv2.imwrite(str(output_file), cropped_image)
        print(f"Cropped object saved to {output_file}")

print("Batch processing finished")
