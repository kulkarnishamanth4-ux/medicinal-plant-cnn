# HerbScan AI 🌿 — Medicinal Plant Identification & Leaf Deficiency Diagnostics

HerbScan AI is an AI-powered Computer Vision web application designed for **Medicinal Plant Species Identification** and **Visual Leaf Health & Deficiency Analysis**. 

Built with **TensorFlow**, **EfficientNetB0**, **OpenCV**, and **Streamlit**, HerbScan AI helps users identify medicinal herbs from leaf images while detecting visual symptoms of key nutrient, mineral, and vitamin deficiencies (e.g., Nitrogen, Phosphorus, Potassium, Magnesium, Iron, Calcium, and Zinc).

---

## 🌟 Key Features

* **🌿 AI Species Classification**: Predicts medicinal plant species across 30+ classes with deep learning Transfer Learning (EfficientNetB0).
* **🧘 Ayurvedic Botanical Profiles**: Automatically links predictions to `plant_database.json`, retrieving scientific names, Ayurvedic properties (*Rasa*, *Guna*, *Virya*, *Vipaka*, *Dosha* pacification), active chemical constituents, and traditional uses.
* **🔬 Visual Leaf Health & Deficiency Analyzer**:
  * **Color Space Decomposition (HSV/LAB)**: Quantifies healthy green tissue, chlorosis (yellowing), necrosis (browning/scorching), and anthocyanin accumulation (purpling).
  * **Interveinal Pattern Extraction**: Detects interveinal chlorosis patterns (common in Iron & Magnesium deficiencies).
  * **Leaf Health Index (0–100%)**: Calculates overall leaf tissue integrity.
  * **Color-Coded Diagnostic Heatmap Overlay**: Visual segmentation mask highlighting healthy vs. damaged leaf zones.
  * **Tailored Organic Remedies**: Provides bio-fertilizer recommendations, Epsom salt foliar sprays, compost teas, and preventive soil management.

### 🔮 Planned Future Enhancements

* **📸 Direct Camera Capture**: Seamlessly open the device camera within the web app to capture and classify leaves on the fly without saving photos locally.
* **🛒 P2P Medicinal Plant Marketplace**: A dedicated ecosystem for users to buy, sell, and trade medicinal plants and seeds.
* **💬 In-App Secure Messaging**: Real-time, authenticated chat rooms for buyers and sellers to communicate, negotiate, and securely share images.
* **🧠 Localized Personal AI Models**: Allow users to fine-tune personal, private versions of the model adapted to the specific geographical variations of plants in their region.
* **🌍 Context-Aware Predictions**: Fuse image data with metadata (geolocation, season, climate) in a multi-modal neural network to drastically improve identification accuracy.

---

## 📁 Repository Structure

```text
├── app.py                             # Interactive Streamlit Web Application
├── deficiency_analyzer.py             # Computer Vision Leaf Health & Deficiency Analyzer
├── deficiency_database.json           # Knowledge Base of Deficiencies, Symptoms & Organic Remedies
├── plant_database.json                # Ayurvedic & Botanical Metadata Database
├── medicinal_leaf_efficientnet.keras  # Pre-trained EfficientNetB0 Keras Model Weights
├── class_names.json                   # Class Label Mappings
├── predict.py                         # CLI Inference & Diagnostic Tool
├── train.py                           # EfficientNet Model Training & Fine-Tuning Pipeline
├── requirements.txt                   # Deployment Dependency Specifications
├── render.yaml                        # Render Cloud Web Service Configuration
└── .streamlit/config.toml             # Streamlit Production Server Configuration
```

---

## 🚀 Quick Start & Local Setup

### 1. Prerequisites & Environment Setup
```bash
# Clone your repository
git clone https://github.com/<your-username>/medicinal-plant-cnn.git
cd medicinal-plant-cnn

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run CLI Inference & Health Analysis
```bash
python predict.py --image your_leaf_sample.jpg
```

---

## ☁️ Deployment Instructions

### Deploying to Render
1. Push your code to your GitHub repository.
2. Log in to [Render](https://render.com) and click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Select **Environment**: `Python 3`.
5. **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
6. **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
7. Click **Create Web Service**.

### Deploying to Streamlit Community Cloud (Recommended Free Hosting)
1. Log in to [Streamlit Cloud](https://share.streamlit.io) using GitHub.
2. Click **New app**.
3. Select your repository, branch (`main`), and set Main file path to `app.py`.
4. Click **Deploy!**
