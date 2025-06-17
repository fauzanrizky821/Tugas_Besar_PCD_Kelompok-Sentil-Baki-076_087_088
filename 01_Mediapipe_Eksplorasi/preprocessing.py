import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def center_and_scale_landmarks(landmarks, nose_idx=1, eye_left_idx=33, eye_right_idx=263):
    """Center landmarks relative to nose and scale based on inter-eye distance."""
    try:
        if landmarks is None or len(landmarks) != 1404:
            logger.error(f"Invalid landmark vector length: {len(landmarks) if landmarks is not None else 'None'}")
            return None
        landmarks = np.array(landmarks).reshape(-1, 3)  # Shape: (468, 3)

        # Center to nose
        nose = landmarks[nose_idx]  # [x, y, z]
        landmarks_centered = landmarks - nose

        # Scale based on inter-eye distance
        eye_left = landmarks[eye_left_idx]
        eye_right = landmarks[eye_right_idx]
        inter_eye_distance = np.linalg.norm(eye_left[:2] - eye_right[:2])  # Distance in x-y plane
        if inter_eye_distance > 0:
            landmarks_scaled = landmarks_centered / inter_eye_distance
        else:
            logger.warning("Inter-eye distance is zero, skipping scaling.")
            landmarks_scaled = landmarks_centered

        return landmarks_scaled.flatten()  # Return as 1404 vector
    except Exception as e:
        logger.error(f"Error in centering and scaling landmarks: {str(e)}")
        return None

def augment_landmarks(landmarks):
    """Apply data augmentation: flip horizontal, add noise, and minor rotation."""
    try:
        if landmarks is None or len(landmarks) != 1404:
            logger.error(f"Invalid landmark vector for augmentation: {len(landmarks) if landmarks is not None else 'None'}")
            return [landmarks] if landmarks is not None else []
        landmarks = np.array(landmarks).reshape(-1, 3)  # Shape: (468, 3)
        augmented = []

        # Original landmarks
        augmented.append(landmarks.flatten())

        # Flip horizontal (negate x-coordinates)
        flipped = landmarks.copy()
        flipped[:, 0] = -flipped[:, 0]
        augmented.append(flipped.flatten())

        # Add noise (±1%)
        noisy = landmarks.copy()
        noise = np.random.uniform(-0.01, 0.01, noisy.shape)
        noisy += noise
        augmented.append(noisy.flatten())

        # Minor rotation (±5°)
        angle = random.uniform(-5, 5) * np.pi / 180  # Convert to radians
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        rotated = np.dot(landmarks, rotation_matrix)
        augmented.append(rotated.flatten())

        return augmented
    except Exception as e:
        logger.error(f"Error in augmenting landmarks: {str(e)}")
        return [landmarks.flatten()] if landmarks is not None else []

def extract_landmarks(image_path, mp_face_mesh):
    """Extract 468 face landmarks from an image using MediaPipe."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            landmark_vector = []
            for lm in landmarks:
                landmark_vector.extend([lm.x, lm.y, lm.z])
            # Center and scale landmarks
            landmark_vector = center_and_scale_landmarks(landmark_vector)
            if landmark_vector is None:
                logger.warning(f"Failed to center/scale landmarks for {image_path}")
                return None
            return landmark_vector
        logger.warning(f"No landmarks detected for {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None
    finally:
        if 'image' in locals():
            del image

def preprocess_all_datasets(raw_csv_dir, image_dir, output_csv):
    """Process all CSV files in raw_csv_dir to extract landmarks and save to output_csv."""
    logger.info("Starting preprocessing...")

    # Initialize MediaPipe Face Mesh
    mp_face = mp.solutions.face_mesh
    mp_face_mesh = None
    try:
        mp_face_mesh = mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
    except Exception as e:
        logger.error(f"Failed to initialize MediaPipe Face Mesh: {str(e)}")
        return

    data = []
    landmark_columns = [f"{coord}{i + 1}" for i in range(468) for coord in ['x', 'y', 'z']]
    columns = landmark_columns + ['age_label', 'expression_label', 'gender_label']

    # Track processing statistics
    stats = {'processed': 0, 'no_landmarks': 0, 'not_found': 0, 'total': 0}

    # Find all CSV files
    csv_files = glob.glob(os.path.join(raw_csv_dir, "*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {raw_csv_dir}")
        return

    for csv_file in csv_files:
        logger.info(f"Processing {csv_file}...")
        try:
            if not os.access(csv_file, os.R_OK):
                logger.error(f"No read permission for {csv_file}")
                continue

            df = pd.read_csv(csv_file, delimiter=',')

            # Check for expected columns
            expected_columns = ['ID', 'AGE', 'EXPRESSION', 'GENDER']
            if not all(col in df.columns for col in expected_columns):
                logger.error(
                    f"{csv_file} missing expected columns. Expected: {expected_columns}, Found: {list(df.columns)}")
                continue

            if df.empty:
                logger.error(f"{csv_file} is empty")
                continue

            stats['total'] += len(df)

            # Process each image
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f"Extracting landmarks from {os.path.basename(csv_file)}"):
                # Try both .jpg and .png extensions
                image_path = os.path.join(image_dir, f"{row['ID']}.jpg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_dir, f"{row['ID']}.png")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_dir, f"{row['ID']}.jpeg")
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    stats['not_found'] += 1
                    continue

                landmarks = extract_landmarks(image_path, mp_face_mesh)

                if landmarks is not None and len(landmarks) == 1404:
                    # Apply augmentation
                    augmented_landmarks = augment_landmarks(landmarks)
                    for aug_landmarks in augmented_landmarks:
                        if aug_landmarks is not None and len(aug_landmarks) == 1404:
                            data.append(aug_landmarks.tolist() + [row['AGE'], row['EXPRESSION'], row['GENDER']])
                            stats['processed'] += 1
                        else:
                            logger.warning(f"Invalid augmented landmarks for {row['ID']}")
                else:
                    stats['no_landmarks'] += 1

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {str(e)}")
            continue

    # Save combined processed data
    if data:
        processed_df = pd.DataFrame(data, columns=columns)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        processed_df.to_csv(output_csv, index=False)
        logger.info(f"Processed data saved to {output_csv}")
    else:
        logger.error("No valid data processed.")

    # Log summary
    logger.info(
        f"Preprocessing Summary: Total entries={stats['total']}, "
        f"Processed={stats['processed']}, No landmarks={stats['no_landmarks']}, "
        f"Not found={stats['not_found']}"
    )

    # Clean up MediaPipe resources
    if mp_face_mesh:
        mp_face_mesh.close()

def normalize_and_split_data(csv_path, output_dir, val_split=0.2, test_split=0.1):
    """Normalize data and split into train/val/test sets."""
    try:
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.error(f"{csv_path} is empty")
            return

        X = df.iloc[:, :-3].values
        y_age = df['age_label'].values
        y_expression = df['expression_label'].values
        y_gender = df['gender_label'].values

        # Z-score normalization
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        from sklearn.preprocessing import LabelEncoder
        le_age = LabelEncoder()
        le_expression = LabelEncoder()
        le_gender = LabelEncoder()

        y_age = le_age.fit_transform(y_age)
        y_expression = le_expression.fit_transform(y_expression)
        y_gender = le_gender.fit_transform(y_gender)

        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_age_train, y_age_temp, y_exp_train, y_exp_temp, y_gen_train, y_gen_temp = train_test_split(
            X, y_age, y_expression, y_gender, test_size=(val_split + test_split), random_state=42
        )
        X_val, X_test, y_age_val, y_age_test, y_exp_val, y_exp_test, y_gen_val, y_gen_test = train_test_split(
            X_temp, y_age_temp, y_exp_temp, y_gen_temp, test_size=test_split / (val_split + test_split), random_state=42
        )

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_age_train.npy'), y_age_train)
        np.save(os.path.join(output_dir, 'y_age_val.npy'), y_age_val)
        np.save(os.path.join(output_dir, 'y_age_test.npy'), y_age_test)
        np.save(os.path.join(output_dir, 'y_exp_train.npy'), y_exp_train)
        np.save(os.path.join(output_dir, 'y_exp_val.npy'), y_exp_val)
        np.save(os.path.join(output_dir, 'y_exp_test.npy'), y_exp_test)
        np.save(os.path.join(output_dir, 'y_gen_train.npy'), y_gen_train)
        np.save(os.path.join(output_dir, 'y_gen_val.npy'), y_gen_val)
        np.save(os.path.join(output_dir, 'y_gen_test.npy'), y_gen_test)

        import pickle
        with open(os.path.join(output_dir, 'le_age.pkl'), 'wb') as f:
            pickle.dump(le_age, f)
            logger.info(f"Saved le_age.pkl with classes: {le_age.classes_}, size: {os.path.getsize(os.path.join(output_dir, 'le_age.pkl'))} bytes")
        with open(os.path.join(output_dir, 'le_expression.pkl'), 'wb') as f:
            pickle.dump(le_expression, f)
            logger.info(f"Saved le_expression.pkl with classes: {le_expression.classes_}, size: {os.path.getsize(os.path.join(output_dir, 'le_expression.pkl'))} bytes")
        with open(os.path.join(output_dir, 'le_gender.pkl'), 'wb') as f:
            pickle.dump(le_gender, f)
            logger.info(f"Saved le_gender.pkl with classes: {le_gender.classes_}, size: {os.path.getsize(os.path.join(output_dir, 'le_gender.pkl'))} bytes")

        logger.info(f"Normalized data and splits saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error in normalization and splitting: {str(e)}")

if __name__ == "__main__":
    raw_csv_dir = "../Dataset/CSV/Raw"
    image_dir = "../Dataset/Image"
    output_csv = "../Dataset/CSV/Processed/processed_dataset.csv"
    output_dir = "../Dataset/Training_Data"

    preprocess_all_datasets(raw_csv_dir, image_dir, output_csv)
    normalize_and_split_data(output_csv, output_dir)