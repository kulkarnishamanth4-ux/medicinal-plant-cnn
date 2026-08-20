import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. Configuration & Labels
import os
BASE_DIR = r"C:\Users\girig\Downloads\medicinal-plant-cnn"
with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)
REGIONS = ["Tropical", "Arid", "Temperate", "Coastal", "Mountain"]
SEASONS = ["Summer", "Winter", "Monsoon", "Spring"]

# Synthetic Priors (Which regions/seasons a plant is MOST likely to be found)
# This simulates actual multi-modal metadata distribution
PRIORS = {
    "Aloe Vera": {"region": ["Arid", "Coastal"], "season": ["Summer", "Spring"]},
    "Amla": {"region": ["Tropical", "Mountain"], "season": ["Winter", "Monsoon"]},
    "Amruthaballi": {"region": ["Tropical", "Coastal"], "season": ["Monsoon", "Summer"]},
    "Arali": {"region": ["Tropical", "Temperate"], "season": ["Summer", "Spring"]},
    "Ashoka Plant": {"region": ["Tropical", "Coastal"], "season": ["Spring", "Summer"]},
    "Ashwagandha": {"region": ["Arid", "Temperate"], "season": ["Winter", "Summer"]},
    "Asthma Weed": {"region": ["Tropical", "Arid"], "season": ["Monsoon", "Spring"]},
    "Balloon Vine": {"region": ["Tropical", "Coastal"], "season": ["Summer", "Monsoon"]},
    "Beans": {"region": ["Temperate", "Tropical"], "season": ["Summer", "Spring"]},
    "Betel": {"region": ["Tropical", "Coastal"], "season": ["Monsoon", "Summer"]},
    "Bhringaraja": {"region": ["Tropical", "Coastal"], "season": ["Monsoon", "Winter"]},
    "Brahmi": {"region": ["Tropical", "Coastal"], "season": ["Monsoon", "Summer"]},
    "Eucalyptus": {"region": ["Temperate", "Mountain"], "season": ["Summer", "Winter"]},
    "Ginger": {"region": ["Tropical", "Coastal"], "season": ["Monsoon", "Summer"]},
    "Mint": {"region": ["Temperate", "Mountain"], "season": ["Spring", "Summer"]},
    "Neem": {"region": ["Tropical", "Arid"], "season": ["Summer", "Spring"]},
    "Rosemary": {"region": ["Temperate", "Coastal"], "season": ["Spring", "Summer"]},
    "Tulsi": {"region": ["Tropical", "Temperate"], "season": ["Monsoon", "Summer"]},
    "Turmeric": {"region": ["Tropical", "Coastal"], "season": ["Monsoon", "Winter"]}
}

# 2. Generate Synthetic Training Data
# We generate simulated outputs of the MobileNet model + Random Environmental context
NUM_SAMPLES_PER_CLASS = 2000
X_cnn = []      # Shape: (N, 19)
X_region = []   # Shape: (N, 5)
X_season = []   # Shape: (N, 4)
y = []          # Shape: (N, 19) (One-hot labels)

print("Generating synthetic multi-modal dataset...")
for i, class_name in enumerate(class_names):
    priors = PRIORS.get(class_name, {"region": REGIONS, "season": SEASONS})
    
    for _ in range(NUM_SAMPLES_PER_CLASS):
        # Simulated CNN output: High probability for the true class, noise for others
        cnn_pred = np.random.uniform(0, 0.2, NUM_CLASSES)
        # Give the true class a dominant score (between 0.4 and 0.9)
        cnn_pred[i] = np.random.uniform(0.4, 0.9)
        # Normalize
        cnn_pred = cnn_pred / np.sum(cnn_pred)
        X_cnn.append(cnn_pred)
        
        # Simulated Metadata: 80% chance it matches the priors, 20% random
        if np.random.rand() < 0.8:
            region = np.random.choice(priors["region"])
            season = np.random.choice(priors["season"])
        else:
            region = np.random.choice(REGIONS)
            season = np.random.choice(SEASONS)
            
        # One-hot encode Region
        reg_vec = np.zeros(len(REGIONS))
        reg_vec[REGIONS.index(region)] = 1
        X_region.append(reg_vec)
        
        # One-hot encode Season
        sea_vec = np.zeros(len(SEASONS))
        sea_vec[SEASONS.index(season)] = 1
        X_season.append(sea_vec)
        
        # Label
        label = np.zeros(NUM_CLASSES)
        label[i] = 1
        y.append(label)

X_cnn = np.array(X_cnn)
X_region = np.array(X_region)
X_season = np.array(X_season)
y = np.array(y)

# 3. Build Multi-Modal Fusion Network
print("Building Late-Fusion Neural Network...")
input_cnn = layers.Input(shape=(NUM_CLASSES,), name="cnn_input")
input_region = layers.Input(shape=(len(REGIONS),), name="region_input")
input_season = layers.Input(shape=(len(SEASONS),), name="season_input")

# Concatenate all features
concat = layers.Concatenate()([input_cnn, input_region, input_season])

# Dense layers to learn correlations between environment and plant features
dense1 = layers.Dense(64, activation='relu')(concat)
dropout = layers.Dropout(0.2)(dense1)
dense2 = layers.Dense(32, activation='relu')(dropout)
output = layers.Dense(NUM_CLASSES, activation='softmax', name="fusion_output")(dense2)

model = models.Model(inputs=[input_cnn, input_region, input_season], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# 4. Train the Model
print("Training the Fusion Network...")
history = model.fit(
    [X_cnn, X_region, X_season], y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 5. Save the Model
save_path = os.path.join(BASE_DIR, "multimodal_fusion.keras")
model.save(save_path)
print(f"✅ Multi-Modal Fusion Model saved as '{save_path}'")
