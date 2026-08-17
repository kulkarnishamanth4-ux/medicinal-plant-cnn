import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

MODEL_PATH = "medicinal_leaf_efficientnet.keras"
IMAGE_PATH = "betel.jpg"

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

image = load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

image = img_to_array(image)
image = np.expand_dims(image, axis=0)

pred = model.predict(image, verbose=0)[0]

order = np.argsort(pred)[::-1]

print("\n==============================")
print("PREDICTIONS")
print("==============================")

for i in order:
    print(f"{class_names[i]:20s} {pred[i] * 100:.2f}%")

print("\nTop prediction:")
print(
    class_names[order[0]],
    f"{pred[order[0]] * 100:.2f}%"
)
