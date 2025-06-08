import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os


def evaluate_model(data_dir, model_path, output_dir):
    """Evaluate the model and generate reports."""
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_age_test = np.load(os.path.join(data_dir, 'y_age_test.npy'))
    y_exp_test = np.load(os.path.join(data_dir, 'y_exp_test.npy'))
    y_gen_test = np.load(os.path.join(data_dir, 'y_gen_test.npy'))

    with open(os.path.join(data_dir, 'le_age.pkl'), 'rb') as f:
        le_age = pickle.load(f)
    with open(os.path.join(data_dir, 'le_expression.pkl'), 'rb') as f:
        le_expression = pickle.load(f)
    with open(os.path.join(data_dir, 'le_gender.pkl'), 'rb') as f:
        le_gender = pickle.load(f)

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(X_test)
    y_age_pred = np.argmax(predictions[0], axis=1)
    y_exp_pred = np.argmax(predictions[1], axis=1)
    y_gen_pred = np.argmax(predictions[2], axis=1)

    print("Age Classification Report:")
    print(classification_report(y_age_test, y_age_pred, target_names=le_age.classes_))
    print("\nExpression Classification Report:")
    print(classification_report(y_exp_test, y_exp_pred, target_names=le_expression.classes_))
    print("\nGender Classification Report:")
    print(classification_report(y_gen_test, y_gen_pred, target_names=le_gender.classes_))

    os.makedirs(output_dir, exist_ok=True)

    cm_age = confusion_matrix(y_age_test, y_age_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_age, annot=True, fmt='d', cmap='Blues', xticklabels=le_age.classes_, yticklabels=le_age.classes_)
    plt.title('Age Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'cm_age.png'))
    plt.close()

    cm_exp = confusion_matrix(y_exp_test, y_exp_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_exp, annot=True, fmt='d', cmap='Blues', xticklabels=le_expression.classes_,
                yticklabels=le_expression.classes_)
    plt.title('Expression Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'cm_expression.png'))
    plt.close()

    cm_gen = confusion_matrix(y_gen_test, y_gen_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_gen, annot=True, fmt='d', cmap='Blues', xticklabels=le_gender.classes_,
                yticklabels=le_gender.classes_)
    plt.title('Gender Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'cm_gender.png'))
    plt.close()

    print(f"Confusion matrices saved to {output_dir}")


if __name__ == "__main__":
    evaluate_model('../Dataset/Model_Output', '../Model/model.h5', '../Evaluasi')