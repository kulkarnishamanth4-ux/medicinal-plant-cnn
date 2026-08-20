import os
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from deficiency_analyzer import analyze_leaf_health

MODEL_PATH = "medicinal_leaf_mobilenetv3.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

def run_prediction(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return

    print("\n==========================================")
    print("🌿 HERBSCAN AI - MODEL & HEALTH DIAGNOSTIC")
    print("==========================================")
    print(f"Loading image: {image_path}")

    # Load Model & Classes
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    # Preprocess Image
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize(IMG_SIZE)
    img_array = np.expand_dims(np.asarray(img_resized, dtype=np.float32), axis=0)

    # 1. Species Inference
    preds = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1]

    print("\n--- SPECIES IDENTIFICATION ---")
    print(f"Top Prediction: {class_names[top_indices[0]]} ({preds[top_indices[0]] * 100:.2f}%)")
    print("\nTop 3 Predictions:")
    for i in range(min(3, len(top_indices))):
        idx = top_indices[i]
        print(f"  {i+1}. {class_names[idx]:20s}: {preds[idx] * 100:.2f}%")

    # 2. Leaf Health & Deficiency Diagnostic
    print("\n--- LEAF HEALTH & DEFICIENCY DIAGNOSIS ---")
    health_res = analyze_leaf_health(img_pil)
    print(f"Overall Leaf Health Score : {health_res['health_score']}%")
    print(f"Primary Condition Detected: {health_res['primary_deficiency']}")
    print(f"Diagnostic Confidence     : {health_res['confidence']}%")
    print("\nColor Metrics Breakdown:")
    for k, v in health_res["metrics"].items():
        print(f"  • {k:20s}: {v}%")
    print("\nIdentified Visual Symptoms:")
    for s in health_res["symptoms"]:
        print(f"  • {s}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HerbScan AI CLI Prediction & Diagnostic Tool")
    parser.add_argument("--image", type=str, default="betel.jpg", help="Path to input leaf image")
    args = parser.parse_args()
    
    run_prediction(args.image)
