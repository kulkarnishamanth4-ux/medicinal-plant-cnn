# HerbScan AI 🌿 — Medicinal Plant Identification & Leaf Deficiency Diagnostics

HerbScan AI is an AI-powered Computer Vision web application designed for **Medicinal Plant Species Identification**, **Visual Leaf Health & Deficiency Analysis**, and **Community-Driven Plant Trading**. 

Built with **TensorFlow**, **EfficientNetB0**, **OpenCV**, **Supabase**, and **Streamlit**, HerbScan AI helps users identify medicinal herbs from leaf images, detect visual symptoms of key nutrient deficiencies, and securely trade plants with other enthusiasts.

---

## 🌟 Key Features

* **🌿 AI Species Classification**: Predicts medicinal plant species across 30+ classes with deep learning Transfer Learning (EfficientNetB0).
* **📸 Direct Camera Capture**: Seamlessly open your device's camera within the web app to capture and classify leaves on the fly.
* **🛒 P2P Medicinal Plant Marketplace**: A dedicated ecosystem for users to buy, sell, and trade medicinal plants, cuttings, and seeds.
* **💬 In-App Secure Messaging**: Real-time, authenticated chat rooms for buyers and sellers to communicate, negotiate, and securely share images.
* **🔒 Enterprise Security**: Fully integrated with Supabase Authentication and strict Row Level Security (RLS) policies to ensure private conversations and secure user data.
* **🧘 Ayurvedic Botanical Profiles**: Automatically links predictions to our database, retrieving scientific names, Ayurvedic properties, active chemical constituents, and traditional uses.
* **🔬 Visual Leaf Health & Deficiency Analyzer**:
  * **Color Space Decomposition (HSV/LAB)**: Quantifies healthy green tissue, chlorosis (yellowing), necrosis (browning/scorching), and anthocyanin accumulation (purpling).
  * **Interveinal Pattern Extraction**: Detects interveinal chlorosis patterns (common in Iron & Magnesium deficiencies).
  * **Leaf Health Index (0–100%)**: Calculates overall leaf tissue integrity.
  * **Color-Coded Diagnostic Heatmap Overlay**: Visual segmentation mask highlighting healthy vs. damaged leaf zones.

### 🔮 Planned Future Enhancements

* **🧠 Localized Personal AI Models**: Allow users to fine-tune personal, private versions of the model adapted to the specific geographical variations of plants in their region.
* **🌍 Context-Aware Predictions**: Fuse image data with metadata (geolocation, season, climate) in a multi-modal neural network to drastically improve identification accuracy.

---

## 📁 Repository Structure

```text
├── app.py                             # Interactive Streamlit Web Application
├── deficiency_analyzer.py             # Computer Vision Leaf Health & Deficiency Analyzer
├── marketplace_db.py                  # Supabase Auth, Marketplace, and Messaging Logic
├── setup_supabase.sql                 # Database Schema and Row Level Security (RLS) Policies
├── security_scan.ps1                  # SAST and SCA Security Scanner 
├── deficiency_database.json           # Knowledge Base of Deficiencies & Organic Remedies
├── plant_database.json                # Ayurvedic & Botanical Metadata Database
├── medicinal_leaf_efficientnet.keras  # Pre-trained EfficientNetB0 Keras Model Weights
├── class_names.json                   # Class Label Mappings
├── predict.py                         # CLI Inference & Diagnostic Tool
├── requirements.txt                   # Dependency Specifications
└── .streamlit/config.toml             # Streamlit Production Server Configuration
```

---

## 🚀 Quick Start & Local Setup

### 1. Prerequisites & Environment Setup
```bash
# Clone your repository
git clone https://github.com/kulkarnishamanth4-ux/medicinal-plant-cnn.git
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

### 2. Configure Supabase Secrets
You must configure your Supabase connection for the Marketplace and Messaging features to work.
Create a file at `.streamlit/secrets.toml` and add your keys:
```toml
[supabase]
url = "https://your-project-id.supabase.co"
key = "your-anon-key"
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
