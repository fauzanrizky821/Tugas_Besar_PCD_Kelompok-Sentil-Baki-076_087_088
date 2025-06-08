Tugas Besar PCD: Facial Expression, Age, and Gender Classification
Overview
This project implements a system for classifying facial expressions (happy, sad, angry, neutral), age categories (young, middle, old), and gender (male, female) using MediaPipe Face Mesh and a multi-output CNN model. It includes a Streamlit interface for real-time detection, dataset addition, and model training.
Directory Structure
Tugas_Besar_PCD_[Nama_Kelompok_NIM]/
├── 01_Mediapipe_Eksplorasi/      # Model training and evaluation scripts
├── 02_Streamlit_Interface/       # Streamlit app
├── Dataset/                      # Images, raw/processed CSVs, model outputs
├── requirements.txt              # Dependencies
├── 03_Dokumen_Proses_Analisis.pdf
├── 04_File_Presentasi.pptx
├── Link_YouTube.txt
├── README.md
└── .gitignore

Setup

Install Python 3.9.13:Download from python.org.

Create Virtual Environment:
python -m venv venv
venv\Scripts\activate  # On Windows


Install Dependencies:
pip install -r requirements.txt


Prepare Dataset:

Place images in Dataset/Image/.
Place raw CSV files (e.g., dataset_1.csv) in Dataset/CSV/Raw/.
CSV format: filename,UMUR,EKS_PRESI,JENIS_KELAMIN.


Preprocess Data:
cd 01_Mediapipe_Eksplorasi
python preprocessing.py


Run Streamlit App:
cd ../02_Streamlit_Interface
streamlit run main.py



Usage

Real-Time Detection: Use webcam for live predictions.
Add Dataset: Capture or upload images, add labels, and save to captured_processed_dataset.csv.
Train Model: Train on combined datasets with progress visualization.

Requirements

Python 3.9.13
Webcam for real-time detection
Sufficient disk space for datasets and model outputs

Team
[Nama_Kelompok_NIM]
