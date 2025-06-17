import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model_path, data_dir, output_dir):
    """Evaluate the model and generate confusion matrices and metrics for age, expression, and gender."""
    logger.info("Starting model evaluation...")

    try:
        # Load model
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load test data
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_age_test = np.load(os.path.join(data_dir, 'y_age_test.npy'))
        y_exp_test = np.load(os.path.join(data_dir, 'y_exp_test.npy'))
        y_gen_test = np.load(os.path.join(data_dir, 'y_gen_test.npy'))

        # Load label encoders with validation
        encoders = {}
        for encoder_file in ['le_age.pkl', 'le_expression.pkl', 'le_gender.pkl']:
            file_path = os.path.join(data_dir, encoder_file)
            if not os.path.exists(file_path):
                logger.error(f"Encoder file not found: {file_path}")
                return None
            try:
                with open(file_path, 'rb') as f:
                    encoders[encoder_file[:-4]] = pickle.load(f)
                logger.info(f"Loaded {encoder_file} with classes: {encoders[encoder_file[:-4]].classes_}")
            except EOFError as e:
                logger.error(f"Encoder file {file_path} is empty or corrupt: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                return None

        le_age = encoders['le_age']
        le_expression = encoders['le_expression']
        le_gender = encoders['le_gender']

        # Get predictions
        age_pred, exp_pred, gen_pred = model.predict(X_test, verbose=0)
        age_pred_labels = np.argmax(age_pred, axis=1)
        exp_pred_labels = np.argmax(exp_pred, axis=1)
        gen_pred_labels = np.argmax(gen_pred, axis=1)

        # Compute confusion matrices
        cm_age = confusion_matrix(y_age_test, age_pred_labels)
        cm_exp = confusion_matrix(y_exp_test, exp_pred_labels)
        cm_gen = confusion_matrix(y_gen_test, gen_pred_labels)

        # Compute additional metrics
        metrics = {
            'age': {
                'accuracy': np.mean(age_pred_labels == y_age_test),
                'f1_score': f1_score(y_age_test, age_pred_labels, average='weighted'),
                'precision': precision_score(y_age_test, age_pred_labels, average='weighted'),
                'recall': recall_score(y_age_test, age_pred_labels, average='weighted')
            },
            'expression': {
                'accuracy': np.mean(exp_pred_labels == y_exp_test),
                'f1_score': f1_score(y_exp_test, exp_pred_labels, average='weighted'),
                'precision': precision_score(y_exp_test, exp_pred_labels, average='weighted'),
                'recall': recall_score(y_exp_test, exp_pred_labels, average='weighted')
            },
            'gender': {
                'accuracy': np.mean(gen_pred_labels == y_gen_test),
                'f1_score': f1_score(y_gen_test, gen_pred_labels, average='weighted'),
                'precision': precision_score(y_gen_test, gen_pred_labels, average='weighted'),
                'recall': recall_score(y_gen_test, gen_pred_labels, average='weighted')
            }
        }

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics to text file
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            for label, scores in metrics.items():
                f.write(f"{label.capitalize()} Metrics:\n")
                for metric, value in scores.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        logger.info(f"Saved metrics to {metrics_path}")

        # Plot and save confusion matrices
        def plot_confusion_matrix(cm, labels, title, filename):
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(filename)
            plt.close()
            logger.info(f"Saved confusion matrix: {filename}")

        # Save confusion matrices
        age_labels = le_age.classes_
        exp_labels = le_expression.classes_
        gen_labels = le_gender.classes_

        cm_age_path = os.path.join(output_dir, 'cm_age.png')
        cm_exp_path = os.path.join(output_dir, 'cm_expression.png')
        cm_gen_path = os.path.join(output_dir, 'cm_gender.png')

        plot_confusion_matrix(cm_age, age_labels, 'Confusion Matrix - Age', cm_age_path)
        plot_confusion_matrix(cm_exp, exp_labels, 'Confusion Matrix - Expression', cm_exp_path)
        plot_confusion_matrix(cm_gen, gen_labels, 'Confusion Matrix - Gender', cm_gen_path)

        logger.info("Model evaluation completed.")
        return {
            'cm_age': cm_age_path,
            'cm_expression': cm_exp_path,
            'cm_gender': cm_gen_path,
            'metrics': metrics_path
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return None

if __name__ == "__main__":
    evaluate_model(
        '../01_Mediapipe_Eksplorasi/Model/model.h5',
        '../Dataset/Training_Data',
        '../01_Mediapipe_Eksplorasi/Evaluasi'
    )