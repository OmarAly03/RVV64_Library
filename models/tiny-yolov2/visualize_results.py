import cv2
import numpy as np
import argparse
import os

def parse_detection_file(detection_file):
    """Parse the detection results text file"""
    detections = []
    
    with open(detection_file, 'r') as f:
        lines = f.readlines()
    
    current_detection = {}
    for line in lines:
        line = line.strip()
        if line.startswith('Detection ') and line.endswith(':'):
            if current_detection:  # Save previous detection
                detections.append(current_detection)
            current_detection = {}
        elif line.startswith('Class: '):
            # Extract class name and ID
            parts = line.split('(ID: ')
            class_name = parts[0].replace('Class: ', '').strip()
            class_id = int(parts[1].replace(')', ''))
            current_detection['class'] = class_name
            current_detection['class_id'] = class_id
        elif line.startswith('Confidence: '):
            current_detection['confidence'] = float(line.replace('Confidence: ', ''))
        elif line.startswith('Center: '):
            # Extract center coordinates: "Center: (1.79536, 5.57569)"
            coords = line.replace('Center: (', '').replace(')', '').split(', ')
            current_detection['center_x'] = float(coords[0])
            current_detection['center_y'] = float(coords[1])
        elif line.startswith('Size: '):
            # Extract width and height: "Size: 3.58178 x 6.42337"
            size = line.replace('Size: ', '').split(' x ')
            current_detection['width'] = float(size[0])
            current_detection['height'] = float(size[1])
        elif line.startswith('Bounding Box: '):
            # Extract corner coordinates: "Bounding Box: (0.00446558, 2.364) to (3.58625, 8.78738)"
            try:
                coords_part = line.replace('Bounding Box: ', '').strip()
                # Split by " to "
                if ' to ' in coords_part:
                    coord_parts = coords_part.split(' to ')
                    # First part: "(0.00446558, 2.364)"
                    min_part = coord_parts[0].replace('(', '').replace(')', '').strip()
                    # Second part: "(3.58625, 8.78738)"
                    max_part = coord_parts[1].replace('(', '').replace(')', '').strip()
                    
                    min_coords = min_part.split(', ')
                    max_coords = max_part.split(', ')
                    
                    current_detection['x_min'] = float(min_coords[0])
                    current_detection['y_min'] = float(min_coords[1])
                    current_detection['x_max'] = float(max_coords[0])
                    current_detection['y_max'] = float(max_coords[1])
                else:
                    print(f"Warning: Could not parse bounding box line: {line}")
            except (IndexError, ValueError) as e:
                print(f"Warning: Error parsing bounding box line '{line}': {e}")
                # Skip this line and continue
                pass
    
    # Don't forget the last detection
    if current_detection:
        detections.append(current_detection)
    
    return detections

def draw_detections(image_path, detections, output_path=None):
    """Draw bounding boxes and labels on the image"""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    print(f"Image dimensions: {img_width}x{img_height}")
    
    # Define colors for different classes (BGR format)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (255, 20, 147), # Deep Pink
    ]
    
    for i, detection in enumerate(detections):
        print(f"Processing detection {i+1}: {detection.get('class', 'unknown')}")
        
        # The coordinates are already in the model's coordinate system (relative to 416x416)
        # We need to scale them to the actual image size
        if 'x_min' in detection and 'x_max' in detection:
            # Scale from model coordinates to image coordinates
            x_min = int(detection['x_min'] * img_width / 13.0)  # Grid is 13x13
            y_min = int(detection['y_min'] * img_height / 13.0)
            x_max = int(detection['x_max'] * img_width / 13.0)
            y_max = int(detection['y_max'] * img_height / 13.0)
            print(f"  Using corner coordinates: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        elif 'center_x' in detection and 'width' in detection:
            # Convert from center coordinates
            center_x = detection['center_x'] * img_width / 13.0
            center_y = detection['center_y'] * img_height / 13.0
            width = detection['width'] * img_width / 13.0
            height = detection['height'] * img_height / 13.0
            
            x_min = int(center_x - width / 2)
            y_min = int(center_y - height / 2)
            x_max = int(center_x + width / 2)
            y_max = int(center_y + height / 2)
            print(f"  Using center coordinates: center=({center_x:.1f}, {center_y:.1f}), size=({width:.1f}x{height:.1f})")
            print(f"  Computed corners: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        else:
            print(f"  Warning: Missing coordinate data for detection {i+1}")
            continue
        
        # Ensure coordinates are within image bounds
        x_min = max(0, min(x_min, img_width - 1))
        y_min = max(0, min(y_min, img_height - 1))
        x_max = max(0, min(x_max, img_width - 1))
        y_max = max(0, min(y_max, img_height - 1))
        
        # Skip if box is invalid
        if x_max <= x_min or y_max <= y_min:
            print(f"  Warning: Invalid bounding box for detection {i+1}")
            continue
        
        # Choose color based on class ID
        color = colors[detection.get('class_id', 0) % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Create label text
        label = f"{detection.get('class', 'unknown')}: {detection.get('confidence', 0.0):.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(image, (x_min, y_min - text_height - baseline - 5), 
                      (x_min + text_width, y_min), color, -1)
        
        # Draw text
        cv2.putText(image, label, (x_min, y_min - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to: {output_path}")
    else:
        cv2.imshow('YOLO Detections', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO detection results')
    parser.add_argument('image_path', help='Path to the original image file')
    parser.add_argument('detection_file', help='Path to the detection results text file')
    parser.add_argument('-o', '--output', help='Output image path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} not found")
        return
    
    if not os.path.exists(args.detection_file):
        print(f"Error: Detection file {args.detection_file} not found")
        return
    
    # Parse detections
    print(f"Parsing detection file: {args.detection_file}")
    detections = parse_detection_file(args.detection_file)
    print(f"Found {len(detections)} detections")
    
    # Debug: print first detection
    if detections:
        print("First detection data:", detections[0])
    
    # Draw and save/display
    output_path = args.output or args.image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
    draw_detections(args.image_path, detections, output_path)

if __name__ == "__main__":
    main()