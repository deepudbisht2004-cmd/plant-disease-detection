
# 🌱 Plant Disease Detection System

## About the Project
Plant Disease Detection is an AI-powered application that identifies plant diseases from leaf images using deep learning.

## Technologies Used
- Python
- TensorFlow
- EfficientNetB0
- Streamlit
- NumPy
- Pillow
- PlantVillage Dataset

## Model Information
- Model: EfficientNetB0
- Image Size: 224 × 224
- Number of Classes: 38
- Dataset: PlantVillage

## Features
- Upload plant leaf images
- Predict plant disease
- Display prediction confidence
- Show top 3 predictions
- Prediction history

## Project Structure
```text
plant-disease-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    ├── plant_disease_final_model.keras
    └── class_names.json
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Disclaimer
This project is developed for educational and research purposes. Predictions should not replace professional agricultural diagnosis.

## Developer
Deepak Bisht
