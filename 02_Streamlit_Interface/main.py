import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle
from PIL import Image
import sys
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add 01_Mediapipe_Eksplorasi to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../01_Mediapipe_Eksplorasi')))
try:
    from train import train_model
    from preprocessing import preprocess_all_datasets, normalize_and_split_data
    from evaluate import evaluate_model
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Initialize MediaPipe
mp_face = mp.solutions.face_mesh
mp_face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@st.cache_resource
def load_model_and_encoders(model_path, encoder_dir):
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None, None, None
        model = tf.keras.models.load_model(model_path)
        with open(os.path.join(encoder_dir, 'le_age.pkl'), 'rb') as f:
            le_age = pickle.load(f)
        with open(os.path.join(encoder_dir, 'le_expression.pkl'), 'rb') as f:
            le_expression = pickle.load(f)
        with open(os.path.join(encoder_dir, 'le_gender.pkl'), 'rb') as f:
            le_gender = pickle.load(f)
        return model, le_age, le_expression, le_gender
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        return None, None, None, None


def extract_landmarks(image, mp_face_mesh):
    """Extract 468 face landmarks from an image."""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            landmark_vector = []
            for lm in landmarks:
                landmark_vector.extend([lm.x, lm.y, lm.z])
            return np.array(landmark_vector), results.multi_face_landmarks[0]
        return None, None
    except Exception as e:
        logger.error(f"Error extracting landmarks: {str(e)}")
        return None, None


def get_bounding_box(image, face_landmarks):
    """Calculate bounding box from landmarks."""
    if not face_landmarks or not face_landmarks.landmark:
        return None
    h, w = image.shape[:2]
    x_coords = [lm.x * w for lm in face_landmarks.landmark]
    y_coords = [lm.y * h for lm in face_landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    padding = 20
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    return x_min, y_min, x_max, y_max


def home_page():
    """Display the homepage with application explanation and workflow."""
    st.title("Selamat Datang di Sistem Analisis Wajah")
    st.markdown("""
        ### Tentang Aplikasi
        Sistem Analisis Wajah adalah aplikasi berbasis kecerdasan buatan yang dirancang untuk menganalisis gambar wajah 
        menggunakan teknologi **MediaPipe Face Mesh** dan model pembelajaran mendalam (deep learning). Aplikasi ini mampu 
        memprediksi **usia** (muda, menengah, tua), **ekspresi wajah** (senang, sedih, marah, netral), serta **jenis kelamin** 
        (pria, wanita) berdasarkan landmark wajah yang diekstraksi dari gambar.

        Aplikasi ini cocok untuk keperluan akademik, penelitian, atau pengembangan teknologi pengenalan wajah. Anda dapat 
        menambahkan dataset baru, memproses data, melatih model, melakukan analisis wajah secara real-time menggunakan 
        webcam, atau menganalisis gambar yang diunggah.

        ### Alur Penggunaan Aplikasi
        Untuk hasil terbaik, ikuti langkah-langkah berikut:
        1. **Tambah Dataset (Add Dataset)**:
           - Mulai dari halaman **Tambah Dataset** untuk menambahkan gambar wajah baru.
           - Anda dapat mengambil gambar menggunakan webcam atau mengunggah file gambar (.jpg atau .png).
           - Berikan label untuk usia, ekspresi, dan jenis kelamin, lalu simpan ke dataset.
           - Data akan disimpan di `captured_processed_dataset.csv` untuk digunakan dalam pelatihan.

        2. **Pratinjau Dataset (Preprocess Datasets)**:
           - Kunjungi halaman **Pratinjau Dataset** untuk memproses file CSV di `Dataset/CSV/Raw/` (misalnya, `dataset.csv`).
           - Proses ini akan mengekstraksi landmark wajah dari gambar di `Dataset/Image/` dan menyimpan hasilnya di 
             `processed_dataset.csv`.
           - Data juga akan dinormalisasi dan dibagi menjadi set pelatihan, validasi, dan pengujian, disimpan di 
             `Dataset/Training_Data/`.

        3. **Latih Model (Train Model)**:
           - Buka halaman **Latih Model** untuk melatih model menggunakan data yang telah diproses.
           - Pantau proses pelatihan melalui grafik loss dan akurasi, serta lihat matriks kebingungan (confusion matrix) 
             untuk mengevaluasi performa model pada usia, ekspresi, dan jenis kelamin.
           - Model yang dilatih akan disimpan di `01_Mediapipe_Eksplorasi/Model/model.h5`.

        4. **Analisis Wajah Real-Time (Real-Time Detection)**:
           - Gunakan halaman **Analisis Wajah Real-Time** untuk memprediksi usia, ekspresi, dan jenis kelamin secara langsung 
             melalui webcam.
           - Hasil prediksi dan landmark wajah akan ditampilkan pada kotak pembatas di sekitar wajah yang terdeteksi.

        5. **Analisis Gambar (Analyze Image)**:
           - Gunakan halaman **Analisis Gambar** untuk mengunggah file gambar (.jpg atau .png) dan mendeteksi fitur wajah.
           - Gambar akan menampilkan landmark wajah, kotak pembatas, dan prediksi usia, ekspresi, dan jenis kelamin, baik pada 
             gambar maupun sebagai teks di bawahnya.

        ### Catatan Penting
        - Pastikan folder `Dataset/Image/` berisi gambar yang sesuai dengan ID di `dataset.csv`.
        - Proses pelatihan membutuhkan data yang cukup (misalnya, 348 entri seperti hasil pratinjau sebelumnya).
        - Jika terjadi error, periksa log di halaman **Pratinjau Dataset** atau pastikan semua dependensi terinstall 
          (`requirements.txt`).
        - Untuk hasil optimal, gunakan gambar wajah dengan pencahayaan baik dan tanpa penghalang.

        **Mulai sekarang dengan menavigasi ke halaman "Tambah Dataset" atau "Analisis Gambar" melalui menu di sisi kiri!**
    """)


def add_dataset():
    st.header("Add New Dataset")
    st.write("Capture an image from the webcam or upload an image, then provide labels to add to the dataset.")

    capture_method = st.radio("Choose capture method:", ("Webcam", "Upload Image"))

    image = None
    if capture_method == "Webcam":
        if st.button("Capture Image"):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    image = frame
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Image", use_column_width=True)
                cap.release()
            else:
                st.error("Could not open webcam.")
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file:
            image = np.array(Image.open(uploaded_file))
            st.image(image, caption="Uploaded Image", use_column_width=True)

    if image is not None:
        age_label = st.selectbox("Age Label", ["YOUNG", "MIDDLE", "OLD"])
        expression_label = st.selectbox("Expression Label", ["HAPPY", "SAD", "ANGRY", "NEUTRAL"])
        gender_label = st.selectbox("Gender Label", ["MALE", "FEMALE"])

        if st.button("Save to Dataset"):
            landmarks, _ = extract_landmarks(image, mp_face_mesh)
            if landmarks is not None:
                processed_csv = '../Dataset/CSV/Processed/processed_dataset.csv'
                if os.path.exists(processed_csv):
                    df_norm = pd.read_csv(processed_csv)
                    mean = df_norm.iloc[:, :-3].values.mean(axis=0)
                    std = df_norm.iloc[:, :-3].values.std(axis=0)
                    landmarks = (landmarks - mean) / std
                else:
                    st.warning("Processed dataset not found. Using raw landmarks.")
                landmark_columns = [f"{coord}{i + 1}" for i in range(468) for coord in ['x', 'y', 'z']]
                data = landmarks.tolist() + [age_label, expression_label, gender_label]
                df = pd.DataFrame([data], columns=landmark_columns + ['age_label', 'expression_label', 'gender_label'])

                output_csv = '../Dataset/CSV/Processed/captured_processed_dataset.csv'
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                if os.path.exists(output_csv):
                    df.to_csv(output_csv, mode='a', header=False, index=False)
                else:
                    df.to_csv(output_csv, index=False)
                st.success(f"Data saved to {output_csv}")
            else:
                st.error("No face detected in the image.")


def preprocess_page():
    st.header("Preprocess Datasets")
    st.write("Process all CSV files in Dataset/CSV/Raw/ and save to Dataset/CSV/Processed/processed_dataset.csv.")

    if st.button("Start Preprocessing"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create a log container and summary placeholder
        log_container = st.empty()
        summary_placeholder = st.empty()
        log_buffer = []

        # Custom logging handler to capture logs and summary
        class StreamlitHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.summary = None

            def emit(self, record):
                msg = self.format(record)
                log_buffer.append(msg)
                log_container.text_area("Preprocessing Logs", "\n".join(log_buffer), height=300)
                # Capture summary log
                if "Preprocessing Summary" in msg:
                    self.summary = msg.split(" - INFO - ")[-1]
                    summary_placeholder.info(f"**Preprocessing Summary**\n\n{self.summary}")

        # Configure logging to Streamlit
        handler = StreamlitHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

        try:
            raw_csv_dir = '../Dataset/CSV/Raw'
            image_dir = '../Dataset/Image'
            output_csv = '../Dataset/CSV/Processed/processed_dataset.csv'
            output_dir = '../Dataset/Training_Data'

            if not os.path.exists(raw_csv_dir) or not os.path.exists(image_dir):
                st.error(f"Required directories not found: {raw_csv_dir} or {image_dir}")
                logger.error(f"Required directories not found: {raw_csv_dir} or {image_dir}")
                return

            # Process CSVs
            logger.info("Starting preprocessing of CSV files...")
            preprocess_all_datasets(raw_csv_dir, image_dir, output_csv)

            # Update progress
            progress_bar.progress(0.5)
            logger.info("Preprocessing complete. Normalizing and splitting data...")

            # Normalize and split
            normalize_and_split_data(output_csv, output_dir)

            progress_bar.progress(1.0)
            st.success(f"Preprocessing completed! Data saved to {output_csv} and splits saved to {output_dir}")

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            st.error(f"Error during preprocessing: {str(e)}")

        finally:
            logger.removeHandler(handler)


def train_model_page():
    st.header("Train Model")
    st.write("Train the model using processed_dataset.csv and captured_processed_dataset.csv.")

    if st.button("Start Training"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        batch_text = st.empty()

        class TrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                status_text.text(f"Epoch {epoch + 1}/{self.params['epochs']} - Starting...")

            def on_batch_end(self, batch, logs=None):
                batch_text.text(f"Batch {batch + 1}/{self.params['steps']} - Loss: {logs['loss']:.4f}")

            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.params['epochs']
                progress_bar.progress(min(progress, 1.0))
                elapsed_time = time.time() - self.epoch_start_time
                status_text.text(
                    f"Epoch {epoch + 1}/{self.params['epochs']} - "
                    f"Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}, "
                    f"Time: {elapsed_time:.2f}s"
                )

        # Train model
        history = train_model(
            '../Dataset/CSV/Processed',
            '../01_Mediapipe_Eksplorasi/Model',
            epochs=50,
            batch_size=32,
            callbacks=[TrainingCallback()]
        )

        if history:
            st.success("Training completed!")
            st.write("### Training History")
            st.line_chart({
                'Loss': history.history['loss'],
                'Validation Loss': history.history['val_loss'],
                'Age Accuracy': history.history['age_output_accuracy'],
                'Expression Accuracy': history.history['exp_output_accuracy'],
                'Gender Accuracy': history.history['gen_output_accuracy']
            })

            # Evaluate model and display confusion matrices
            st.write("### Confusion Matrices")
            evaluation_results = evaluate_model(
                '../01_Mediapipe_Eksplorasi/Model/model.h5',
                '../Dataset/Training_Data',
                '../01_Mediapipe_Eksplorasi/Evaluasi'
            )

            if evaluation_results:
                st.success("Evaluation completed! Displaying confusion matrices...")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if os.path.exists(evaluation_results['cm_age']):
                        st.image(evaluation_results['cm_age'], caption="Confusion Matrix - Age", use_column_width=True)
                    else:
                        st.error(f"Age confusion matrix not found: {evaluation_results['cm_age']}")
                with col2:
                    if os.path.exists(evaluation_results['cm_expression']):
                        st.image(evaluation_results['cm_expression'], caption="Confusion Matrix - Expression",
                                 use_column_width=True)
                    else:
                        st.error(f"Expression confusion matrix not found: {evaluation_results['cm_expression']}")
                with col3:
                    if os.path.exists(evaluation_results['cm_gender']):
                        st.image(evaluation_results['cm_gender'], caption="Confusion Matrix - Gender",
                                 use_column_width=True)
                    else:
                        st.error(f"Gender confusion matrix not found: {evaluation_results['cm_gender']}")
            else:
                st.error("Evaluation failed. Check dataset files or logs.")
        else:
            st.error("Training failed. Check dataset files or logs.")


def real_time_detection():
    st.header("Real-Time Face Analysis")
    st.write("Click the button to start/stop webcam for real-time detection.")

    model_path = '../01_Mediapipe_Eksplorasi/Model/model.h5'
    encoder_dir = '../Dataset/Training_Data'
    model, le_age, le_expression, le_gender = load_model_and_encoders(model_path, encoder_dir)
    if model is None:
        return

    # Load normalization parameters
    processed_csv = '../Dataset/CSV/Processed/processed_dataset.csv'
    if os.path.exists(processed_csv):
        df = pd.read_csv(processed_csv)
        mean = df.iloc[:, :-3].values.mean(axis=0)
        std = df.iloc[:, :-3].values.std(axis=0)
    else:
        st.error("Processed dataset not found for normalization.")
        return

    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False

    if st.button("Start/Stop Webcam"):
        st.session_state.webcam_active = not st.session_state.webcam_active

    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    while st.session_state.webcam_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame.")
            break

        landmarks, face_landmarks = extract_landmarks(frame, mp_face_mesh)

        if landmarks is not None:
            # Draw face landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Normalize landmarks and predict
            landmarks = (landmarks - mean) / std
            landmarks = landmarks.reshape(1, -1)

            age_pred, exp_pred, gen_pred = model.predict(landmarks, verbose=0)
            age_label = le_age.inverse_transform([np.argmax(age_pred, axis=1)[0]])[0]
            exp_label = le_expression.inverse_transform([np.argmax(exp_pred, axis=1)[0]])[0]
            gen_label = le_gender.inverse_transform([np.argmax(gen_pred, axis=1)[0]])[0]

            bbox = get_bounding_box(frame, face_landmarks)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {age_label}", (x_min, y_min - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                            2)
                cv2.putText(frame, f"Expression: {exp_label}", (x_min, y_min - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                cv2.putText(frame, f"Gender: {gen_label}", (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()


def analyze_image():
    st.header("Analyze Image")
    st.write("Upload an image to detect facial features (age, expression, gender) and display landmarks.")

    model_path = '../01_Mediapipe_Eksplorasi/Model/model.h5'
    encoder_dir = '../Dataset/Training_Data'
    model, le_age, le_expression, le_gender = load_model_and_encoders(model_path, encoder_dir)
    if model is None:
        return

    # Load normalization parameters
    processed_csv = '../Dataset/CSV/Processed/processed_dataset.csv'
    if os.path.exists(processed_csv):
        df = pd.read_csv(processed_csv)
        mean = df.iloc[:, :-3].values.mean(axis=0)
        std = df.iloc[:, :-3].values.std(axis=0)
    else:
        st.error("Processed dataset not found for normalization.")
        return

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        # Read and display uploaded image
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process image
        landmarks, face_landmarks = extract_landmarks(image, mp_face_mesh)

        if landmarks is not None:
            # Draw face landmarks
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Normalize landmarks and predict
            landmarks = (landmarks - mean) / std
            landmarks = landmarks.reshape(1, -1)

            age_pred, exp_pred, gen_pred = model.predict(landmarks, verbose=0)
            age_label = le_age.inverse_transform([np.argmax(age_pred, axis=1)[0]])[0]
            exp_label = le_expression.inverse_transform([np.argmax(exp_pred, axis=1)[0]])[0]
            gen_label = le_gender.inverse_transform([np.argmax(gen_pred, axis=1)[0]])[0]

            bbox = get_bounding_box(image, face_landmarks)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, f"Age: {age_label}", (x_min, y_min - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                            2)
                cv2.putText(image, f"Expression: {exp_label}", (x_min, y_min - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                cv2.putText(image, f"Gender: {gen_label}", (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

            # Display processed image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Processed Image with Landmarks and Predictions", use_column_width=True)

            # Display prediction text below the image
            st.markdown(f"""
                **Detected Features:**
                ```
                AGE: {age_label}
                EXPRESSION: {exp_label}
                GENDER: {gen_label}
                ```
            """)
        else:
            st.error("No face detected in the image.")


def main():
    st.title("Facial Analysis System")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page",
                                ["Home", "Add Dataset", "Preprocess Datasets", "Train Model", "Real-Time Detection",
                                 "Analyze Image"])

    if page == "Home":
        home_page()
    elif page == "Add Dataset":
        add_dataset()
    elif page == "Preprocess Datasets":
        preprocess_page()
    elif page == "Train Model":
        train_model_page()
    elif page == "Real-Time Detection":
        real_time_detection()
    elif page == "Analyze Image":
        analyze_image()

    mp_face_mesh.close()


if __name__ == "__main__":
    main()