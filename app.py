from flask import Flask, render_template, request
import os
import cv2
import math
import random
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
IMAGE_FOLDER = 'static/images'
MODEL_FOLDER = 'model'


# Helper Function
def enhance(image):
    CLIP_LIMIT = 2.0
    TILE_GRID_SIZE = (16, 32)

    # Random Gamma Correction
    random_gamma = random.uniform(0.5, 1.0)
    gamma_table = np.array([((i / 255.0) ** random_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_image = cv2.LUT(image, gamma_table)

    # BGR to YCrCb
    ycrcb_image = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2YCrCb)

    # CLAHE on Luminance channel
    y, cr, cb = cv2.split(ycrcb_image)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    clahe_y = clahe.apply(y)
    clahe_ycrcb = cv2.merge([clahe_y, cr, cb])

    # Back to BGR
    final_image = cv2.cvtColor(clahe_ycrcb, cv2.COLOR_YCrCb2BGR)
    return final_image

def count_objects(result):
    names = result[0].names
    cls_list = [names[int(cls)] for cls in result[0].boxes.cls.tolist()]
    return {
        "BUS": cls_list.count("bus"),
        "CAR": cls_list.count("car"),
        "MOTOR": cls_list.count("motor"),
        "TRUCK": cls_list.count("truck")
    }

def calculate_entropy(image: np.ndarray) -> float:
    if image is None:
        return 0.0
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    histogram_length = sum(histogram)
    if histogram_length == 0:
        return 0.0
    samples_probability = [float(h) / histogram_length for h in histogram if h != 0]
    return -sum([p * math.log(p, 2) for p in samples_probability])

def calculate_cii(original_image: np.ndarray, enhanced_image: np.ndarray) -> float:
    if original_image is None or enhanced_image is None:
        return 0.0
    std_dev_orig = np.std(original_image)
    std_dev_enhanced = np.std(enhanced_image)
    if std_dev_orig == 0:
        return float('inf')
    return std_dev_enhanced / std_dev_orig

@app.route("/")
def home():
    return render_template(
        "index.html", 
        upload_filename=None,
        yolo_filename=None,
        clahe_filename=None)

@app.route("/", methods=["POST"])
def detect():
    # Input Image
    image = request.files["image"]
    _, ext = os.path.splitext(image.filename)
    upload_filename = f"upload{ext}"
    
    # Save to static/images
    save_path = os.path.join(IMAGE_FOLDER, upload_filename)
    image.save(save_path)

    ##########################################################
    img = cv2.imread(save_path)

    # Enhance image
    enhanced_image = enhance(img)
    enhanced_path = os.path.join(IMAGE_FOLDER, f"enhanced{ext}")
    cv2.imwrite(enhanced_path, enhanced_image)

    # Apply YOLOv10n
    # Without enhancements
    yolo_model = YOLO(os.path.join(MODEL_FOLDER, 'yolov10n.pt'))
    results_original = yolo_model.predict(save_path) 
    cv2.imwrite(os.path.join(IMAGE_FOLDER, f"result_original{ext}"), results_original[0].plot())


    # With enhancements
    yolo_clahe_model = YOLO(os.path.join(MODEL_FOLDER, 'best.pt'))
    results_enhanced = yolo_clahe_model.predict(enhanced_path)
    cv2.imwrite(os.path.join(IMAGE_FOLDER, f"result_enhanced{ext}"), results_enhanced[0].plot())


    # Count classes
    counts_original = count_objects(results_original)
    counts_enhanced = count_objects(results_enhanced)

    # Metrics
    # Entropy
    entropy_original = round(calculate_entropy(img), 3)
    entropy_enhanced = round(calculate_entropy(enhanced_image), 3)

    # CII
    cii_value = round(calculate_cii(img, enhanced_image), 3)

    
    # Render
    return render_template(
        "index.html", 
        upload_filename=upload_filename,
        yolo_filename=f"result_original{ext}",
        clahe_filename=f"result_enhanced{ext}",
        counts_original=counts_original,
        counts_enhanced=counts_enhanced,
        entropy_original=entropy_original,
        entropy_enhanced=entropy_enhanced,
        cii=cii_value
        )

if __name__ == "__main__":
    app.run(debug=False)
