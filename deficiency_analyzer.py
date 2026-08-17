import cv2
import numpy as np
from PIL import Image

def analyze_leaf_health(image_pil):
    """
    Analyzes a PIL leaf image for visual health symptoms, color breakdown,
    chlorosis/necrosis/purpling ratios, interveinal chlorosis patterns,
    and returns diagnostic metrics and a visual diagnostic heatmap mask.
    """
    # Convert PIL Image to RGB OpenCV array
    img_rgb = np.array(image_pil.convert("RGB"))
    height, width, _ = img_rgb.shape
    total_pixels = height * width

    # Convert RGB to HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # 1. Background vs Leaf Masking (Otsu thresholding + saturation/value bounds)
    leaf_mask = (s > 25) & (v > 20) & (v < 245)
    leaf_pixel_count = np.count_nonzero(leaf_mask)

    if leaf_pixel_count == 0:
        leaf_mask = np.ones((height, width), dtype=bool)
        leaf_pixel_count = total_pixels

    # 2. Color Region Masks (within leaf area)
    green_mask = leaf_mask & (h >= 35) & (h <= 85) & (s > 30)
    yellow_mask = leaf_mask & (((h >= 20) & (h < 35) & (s > 35)) | ((h >= 35) & (h <= 42) & (s <= 60)))
    brown_mask = leaf_mask & (((h >= 8) & (h < 20) & (s > 40) & (v < 180)) | ((v < 60) & (s < 50)))
    purple_mask = leaf_mask & (((h < 10) | (h > 155)) & (s > 40))

    # Pixel counts & ratios
    green_count = np.count_nonzero(green_mask)
    yellow_count = np.count_nonzero(yellow_mask)
    brown_count = np.count_nonzero(brown_mask)
    purple_count = np.count_nonzero(purple_mask)

    green_ratio = green_count / leaf_pixel_count
    yellow_ratio = yellow_count / leaf_pixel_count
    brown_ratio = brown_count / leaf_pixel_count
    purple_ratio = purple_count / leaf_pixel_count

    # 3. Interveinal Chlorosis Detection
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()
    
    interveinal_score = 0.0
    if yellow_ratio > 0.10 and green_ratio > 0.15:
        interveinal_score = float(min(1.0, (yellow_ratio / (green_ratio + 1e-5)) * (laplacian_var / 500.0)))

    # 4. Health Score Calculation (0 to 100%)
    health_score = max(0.0, min(100.0, (green_ratio * 100.0) - (yellow_ratio * 40.0) - (brown_ratio * 75.0) - (purple_ratio * 50.0)))
    
    if green_ratio > 0.70 and brown_ratio < 0.05 and yellow_ratio < 0.10:
        health_score = max(health_score, 88.0)

    # 5. Diagnostic Feature Matrix & Classification Rule Engine
    detected_deficiency = "Healthy"
    confidence_score = 90.0
    visual_symptoms = []

    if brown_ratio > 0.12 or (brown_ratio > 0.06 and yellow_ratio > 0.15):
        detected_deficiency = "Potassium (K)"
        confidence_score = min(96.0, 70.0 + brown_ratio * 100)
        visual_symptoms.append("Marginal leaf scorch and necrotic browning along edges")
        if yellow_ratio > 0.10:
            visual_symptoms.append("Chlorosis progressing inward from leaf margins")

    elif purple_ratio > 0.08:
        detected_deficiency = "Phosphorus (P)"
        confidence_score = min(95.0, 75.0 + purple_ratio * 120)
        visual_symptoms.append("Dark reddish-purple discoloration on leaf tissue")
        visual_symptoms.append("Stunted chlorophyll development and anthocyanin accumulation")

    elif yellow_ratio > 0.18 and interveinal_score > 0.25:
        if yellow_ratio > 0.35:
            detected_deficiency = "Iron (Fe)"
            confidence_score = min(94.0, 72.0 + yellow_ratio * 60)
            visual_symptoms.append("Severe interveinal chlorosis (yellowing between dark green veins)")
            visual_symptoms.append("Fading lamina with prominent vascular pattern")
        else:
            detected_deficiency = "Magnesium (Mg)"
            confidence_score = min(92.0, 70.0 + yellow_ratio * 65)
            visual_symptoms.append("Interveinal chlorosis starting on mature leaves")
            visual_symptoms.append("Green veins remaining with pale yellow interveinal tissue")

    elif yellow_ratio > 0.20:
        detected_deficiency = "Nitrogen (N)"
        confidence_score = min(95.0, 70.0 + yellow_ratio * 70)
        visual_symptoms.append("General pale green to yellow chlorosis across leaf surface")
        visual_symptoms.append("Reduced chlorophyll density and uniform yellowing")

    elif brown_ratio > 0.07:
        detected_deficiency = "Calcium (Ca)"
        confidence_score = min(88.0, 65.0 + brown_ratio * 150)
        visual_symptoms.append("Localized tip burn and leaf margin distortion")

    elif yellow_ratio > 0.12:
        detected_deficiency = "Zinc (Zn)"
        confidence_score = min(85.0, 65.0 + yellow_ratio * 80)
        visual_symptoms.append("Mottled chlorosis and irregular yellow patches")

    else:
        detected_deficiency = "Healthy"
        confidence_score = max(85.0, health_score)
        visual_symptoms.append("Uniform green leaf pigmentation")
        visual_symptoms.append("No significant chlorosis, necrosis, or purpling detected")

    # 6. Generate Color-Coded Diagnostic Heatmap Overlay
    heatmap = np.zeros_like(img_rgb)
    heatmap[green_mask] = [34, 177, 76]      # Green
    heatmap[yellow_mask] = [255, 201, 14]     # Yellow
    heatmap[brown_mask] = [237, 28, 36]       # Red-Brown
    heatmap[purple_mask] = [163, 73, 164]     # Purple

    blended = cv2.addWeighted(img_rgb, 0.70, heatmap, 0.30, 0)
    overlay_pil = Image.fromarray(blended)

    return {
        "health_score": round(health_score, 1),
        "primary_deficiency": detected_deficiency,
        "confidence": round(confidence_score, 1),
        "symptoms": visual_symptoms,
        "metrics": {
            "green_pct": round(green_ratio * 100, 1),
            "chlorosis_pct": round(yellow_ratio * 100, 1),
            "necrosis_pct": round(brown_ratio * 100, 1),
            "purpling_pct": round(purple_ratio * 100, 1),
            "interveinal_index": round(interveinal_score * 100, 1)
        },
        "diagnostic_heatmap": overlay_pil
    }
