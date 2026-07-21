import os
import cv2
import numpy as np

# Directory for preset images
PRESETS_DIR = 'images'
STATIC_DIR = os.path.join('static', 'uploads')
BACKGROUND_PATH = os.path.join('static', 'background_cache.jpg')

def get_presets_background():
    """Compute and cache the median background for preset images to enable subtraction."""
    if os.path.exists(BACKGROUND_PATH):
        bg = cv2.imread(BACKGROUND_PATH)
        if bg is not None:
            return bg
            
    if not os.path.exists(PRESETS_DIR):
        return None
        
    preset_files = sorted([f for f in os.listdir(PRESETS_DIR) if f.endswith('.jpg')])
    if len(preset_files) < 3:
        return None
        
    frames = []
    for f in preset_files:
        path = os.path.join(PRESETS_DIR, f)
        img = cv2.imread(path)
        if img is not None:
            frames.append(img)
            
    if not frames:
        return None
        
    # Temporal median filter
    frames_arr = np.stack(frames, axis=0)
    median_frame = np.median(frames_arr, axis=0).astype(np.uint8)
    
    os.makedirs(os.path.dirname(BACKGROUND_PATH), exist_ok=True)
    cv2.imwrite(BACKGROUND_PATH, median_frame)
    return median_frame

def detect_preset_image(image_path, output_annotated_path):
    """Detect moving vehicles using temporal background subtraction (highly accurate for presets)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    bg = get_presets_background()
    if bg is None or bg.shape != img.shape:
        # Fallback to single image detection if background template cannot be computed
        return detect_single_image(image_path, output_annotated_path)
        
    # Subtract background
    diff = cv2.absdiff(img, bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Threshold and morph
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vehicle_count = 0
    annotated = img.copy()
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter contours by typical car sizes in the scene
        if w > 20 and h > 20 and w < 200 and h < 200:
            vehicle_count += 1
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (46, 204, 113), 2)
            cv2.putText(annotated, f"Vehicle {vehicle_count}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (46, 204, 113), 1)
            
    os.makedirs(os.path.dirname(output_annotated_path), exist_ok=True)
    cv2.imwrite(output_annotated_path, annotated)
    
    return vehicle_count

def detect_single_image(image_path, output_annotated_path):
    """Detect vehicles in a single custom image using edge/contour analysis (fallback/custom upload)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding or Canny edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to merge car features together
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vehicle_count = 0
    annotated = img.copy()
    
    # Limit search to the road section (usually bottom 70% of the image)
    height, width, _ = img.shape
    road_min_y = int(height * 0.25)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Check if the blob is within typical vehicle aspect ratio and area, and on the road
        if y > road_min_y:
            if 22 < w < 220 and 20 < h < 200 and 0.4 < w/h < 2.5:
                vehicle_count += 1
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (46, 204, 113), 2)
                cv2.putText(annotated, f"Vehicle {vehicle_count}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (46, 204, 113), 1)
                
    os.makedirs(os.path.dirname(output_annotated_path), exist_ok=True)
    cv2.imwrite(output_annotated_path, annotated)
    
    return vehicle_count

def detect_vehicles(image_path, output_annotated_path):
    """
    Main detection entry point. Determines whether to use preset-median subtraction or single-image contours.
    """
    # Check if the file is one of the presets
    filename = os.path.basename(image_path)
    is_preset = False
    if os.path.exists(os.path.join(PRESETS_DIR, filename)):
        is_preset = True
        
    if is_preset:
        vehicle_count = detect_preset_image(image_path, output_annotated_path)
    else:
        vehicle_count = detect_single_image(image_path, output_annotated_path)
        
    # Estimate metrics matching the training distribution:
    # Density: ranges from 10 to 200 in the dataset
    density = int(min(200, max(10, vehicle_count * 14 + np.random.randint(-4, 4))))
    
    # Speed: ranges from 15 to 95, inversely proportional to density
    speed = max(15.0, min(95.0, 92.0 - 0.42 * density + np.random.uniform(-4, 4)))
    
    # Flow = density * speed * constant
    flow = max(10.0, density * speed * 0.15 + np.random.uniform(-8, 8))
    
    return {
        "vehicle_count": int(vehicle_count),
        "density": int(density),
        "speed": float(round(speed, 2)),
        "flow": float(round(flow, 2))
    }

if __name__ == '__main__':
    # Test background computation
    bg = get_presets_background()
    if bg is not None:
        print("Median background successfully cached.")
    else:
        print("Failed to compute background cache.")
