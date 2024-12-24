import cv2
import random
from pathlib import Path


input_dir = Path('/home/sdowell1/1_finalProject/yolov5/cropped_roofs')
output_dir = Path('anomaly_images_trial1')  
output_dir.mkdir(exist_ok=True)


DOT_RADIUS = 10 
DOT_COLOR = (0, 0, 255)  
DOT_THICKNESS = -1  

for image_path in input_dir.glob('*.jpg'):  
    print(f"Processing: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        continue

    height, width, _ = image.shape

    center_x = random.randint(DOT_RADIUS, width - DOT_RADIUS)
    center_y = random.randint(DOT_RADIUS, height - DOT_RADIUS)
    cv2.circle(image, (center_x, center_y), DOT_RADIUS, DOT_COLOR, DOT_THICKNESS)
    output_file = output_dir / image_path.name
    cv2.imwrite(str(output_file), image)
    print(f"Saved image with dot to: {output_file}")

print("Processing finished")
