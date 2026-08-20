# HerbScan AI 🌿 — Medicinal Plant Identification & Leaf Deficiency Diagnostics

HerbScan AI is an AI-powered Computer Vision web application designed for **Medicinal Plant Species Identification**, **Visual Leaf Health & Deficiency Analysis**, and **Community-Driven Plant Trading**. 

Built with **TensorFlow**, **MobileNetV3Large**, **OpenCV**, **Supabase**, and **Streamlit**, HerbScan AI helps users identify medicinal herbs from leaf images, detect visual symptoms of key nutrient deficiencies, and securely trade plants with other enthusiasts.

---

## 🌟 Key Features

* **🌿 AI Species Classification**: Predicts medicinal plant species across 19 unique classes (trained on 100 high-quality images per class) with lightweight, high-speed deep learning (MobileNetV3Large).
* **📸 Direct Camera Capture**: Seamlessly open your device's camera within the web app to capture and classify leaves on the fly.
* **🛒 P2P Medicinal Plant Marketplace**: A dedicated ecosystem for users to buy, sell, and trade medicinal plants, cuttings, and seeds.
* **💬 In-App Secure Messaging**: Real-time, authenticated chat rooms for buyers and sellers to communicate, negotiate, and securely share images.
* **🔒 Enterprise Security**: Fully integrated with Supabase Authentication and strict Row Level Security (RLS) policies to ensure private conversations and secure user data.
* **🔊 Voice Pronunciation**: Dynamic Text-to-Speech (TTS) integration that reads out plant scientific and Sanskrit names.
* **📝 Multi-Modal Metadata**: Intercepts and records environmental context (Season, Soil, Region) for unknown plants to build future Vision-Language datasets.
* **🧘 Ayurvedic Botanical Profiles**: Automatically links predictions to our database, retrieving scientific names, Ayurvedic properties, active chemical constituents, and traditional uses.
* **🔬 Visual Leaf Health & Deficiency Analyzer**:
  * **Color Space Decomposition (HSV/LAB)**: Quantifies healthy green tissue, chlorosis (yellowing), necrosis (browning/scorching), and anthocyanin accumulation (purpling).
  * **Interveinal Pattern Extraction**: Detects interveinal chlorosis patterns (common in Iron & Magnesium deficiencies).
  * **Leaf Health Index (0–100%)**: Calculates overall leaf tissue integrity.
  * **Color-Coded Diagnostic Heatmap Overlay**: Visual segmentation mask highlighting healthy vs. damaged leaf zones.
* **🟢 Zero-Latency Edge AI (AR Scanner)**: A custom WebRTC camera integration that runs `TensorFlow.js` entirely in the user's browser, enabling real-time 30FPS inference without any server latency.
* **🪴 My Digital Garden**: A personal, private inventory where users can save their scanned plants, track their health status, and add custom notes.
* **🗺️ Live Foraging Map**: A community-driven interactive globe where users can drop GPS pins of medicinal plants they discover in the wild.
* **🧠 Decentralized Federated Learning**: An advanced browser-based AI trainer. Users can teach the AI entirely new plant species using local KNN embeddings, and securely sync those mathematical weights to a global database for other users to download instantly.

* **🌍 Context-Aware Predictions (Late-Fusion)**: Fuses image data with environmental metadata (Region, Season) in a secondary Multi-Modal Neural Network to drastically improve identification accuracy.
* **🌲 Deep Forest Mode (Offline PWA)**: A standalone Progressive Web App with a Service Worker that caches the AI model and UI to the user's phone. Works 100% offline using IndexedDB, allowing users to scan and save plants deep in the forest, then bulk-sync to the cloud when Wi-Fi is restored.
* **🔄 Continuous Learning (Data Flywheel)**: Automates model improvement by allowing users to push verified field captures to a Supabase community dataset, simulating a production data-flywheel architecture.
* **🗣️ Vernacular AI (Regional Translations)**: Breaks language barriers by dynamically translating scientific nomenclature and Ayurvedic benefits into Indian regional languages (Hindi, Kannada, Tamil, Telugu, Malayalam) with generated Text-to-Speech audio.
* **⛓️ Blockchain Supply Chain Provenance**: Protects marketplace authenticity by generating immutable SHA-256 cryptographic block hashes for every medicinal plant listing, ensuring seed-to-shelf traceability.
* **🩺 eSanjeevani Telemedicine Integration**: Bridges AI diagnostics with medical safety by providing a simulated booking portal for users to consult certified AYUSH doctors before consuming wild plants.
* **🧊 WebXR 3D Molecular Overlays**: Uses Google's `<model-viewer>` component to render fully interactive, rotatable 3D WebXR models of the plant's active chemical constituents directly in the browser.

### 🔮 Planned Future Enhancements
* **🧱 3D Reconstruction**: Photogrammetry implementation to provide 3D visual cues for the plants.

---

## 📁 Repository Structure

```text
├── app.py                             # Interactive Streamlit Web Application
├── ar_scanner.html                    # Custom WebRTC Edge AI Scanner Component
├── offline_scanner.html               # Deep Forest Offline PWA Scanner Interface
├── sw.js                              # PWA Service Worker for Offline Caching
├── manifest.json                      # PWA Installation Manifest
├── federated_trainer.html             # Browser-based Federated Learning KNN Trainer
├── deficiency_analyzer.py             # Computer Vision Leaf Health & Deficiency Analyzer
├── marketplace_db.py                  # Supabase Auth, Marketplace, and Messaging Logic
├── garden_map_db.py                   # Digital Garden & Foraging Map Database Wrapper
├── data_flywheel_db.py                # Supabase wrapper for Community Dataset
├── continuous_learning.py             # Automated Script simulating Model Fine-Tuning
├── federated_db.py                    # Federated Learning Synchronization Logic
├── setup_supabase.sql                 # Database Schema and Row Level Security (RLS) Policies
├── security_scan.ps1                  # SAST and SCA Security Scanner 
├── deficiency_database.json           # Knowledge Base of Deficiencies & Organic Remedies
├── plant_database.json                # Ayurvedic & Botanical Metadata Database
├── medicinal_leaf_mobilenetv3.keras   # Pre-trained MobileNetV3Large Keras Model Weights
├── multimodal_fusion.keras            # Late-Fusion Multi-Modal Neural Network
├── train_multimodal.py                # Synthetic Data Generator & Fusion Training Script
├── tfjs_model/                        # TensorFlow.js GraphModel for Edge Inference
├── train_mobilenet.ipynb              # Google Colab Training Script
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
