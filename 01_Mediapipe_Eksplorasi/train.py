import pickle

import numpy as np
import tensorflow as tf
import pandas as pd
import os
from model import create_cnn_model


def train_model(data_dir, output_dir, epochs=50, batch_size=32, callbacks=None):
    """Train the multi-output CNN model on processed datasets."""
    # Load data from both processed_dataset.csv and captured_processed_dataset.csv
    processed_csv = os.path.join(data_dir, 'processed_dataset.csv')
    captured_csv = os.path.join(data_dir, 'captured_processed_dataset.csv')

    dfs = []
    for csv_file in [processed_csv, captured_csv]:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            dfs.append(df)
        else:
            print(f"Warning: {csv_file} not found.")

    if not dfs:
        print("Error: No valid datasets found.")
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Prepare data
    X = df.iloc[:, :-3].values
    y_age = df['age_label'].values
    y_expression = df['expression_label'].values
    y_gender = df['gender_label'].values

    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Load label encoders
    from sklearn.preprocessing import LabelEncoder
    le_age = LabelEncoder()
    le_expression = LabelEncoder()
    le_gender = LabelEncoder()

    y_age = le_age.fit_transform(y_age)
    y_expression = le_expression.fit_transform(y_expression)
    y_gender = le_gender.fit_transform(y_gender)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_age_train, y_age_val, y_exp_train, y_exp_val, y_gen_train, y_gen_val = train_test_split(
        X, y_age, y_expression, y_gender, test_size=0.2, random_state=42
    )

    # Create model
    model = create_cnn_model()

    # Train model
    history = model.fit(
        X_train,
        {'age_output': y_age_train, 'exp_output': y_exp_train, 'gen_output': y_gen_train},
        validation_data=(X_val, {'age_output': y_age_val, 'exp_output': y_exp_val, 'gen_output': y_gen_val}),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,  # Suppress default output for custom callback
        callbacks=callbacks or []
    )

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'model.h5'))

    # Save new label encoders
    with open(os.path.join(data_dir, 'le_age.pkl'), 'wb') as f:
        pickle.dump(le_age, f)
    with open(os.path.join(data_dir, 'le_expression.pkl'), 'wb') as f:
        pickle.dump(le_expression, f)
    with open(os.path.join(data_dir, 'le_gender.pkl'), 'wb') as f:
        pickle.dump(le_gender, f)

    print(f"Model saved to {output_dir}/model.h5")
    return history


if __name__ == "__main__":
    train_model('../Dataset/Training_Data', '../01_Mediapipe_Eksplorasi/Model')