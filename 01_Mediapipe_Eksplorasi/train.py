import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from model import create_mlp_model
from tensorflow.keras.callbacks import EarlyStopping

def train_model(data_dir, output_dir, epochs=50, batch_size=32, callbacks=None):
    """Train the multi-output MLP model on processed datasets."""
    # Load data from NumPy arrays
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        y_age_train = np.load(os.path.join(data_dir, 'y_age_train.npy'))
        y_age_val = np.load(os.path.join(data_dir, 'y_age_val.npy'))
        y_exp_train = np.load(os.path.join(data_dir, 'y_exp_train.npy'))
        y_exp_val = np.load(os.path.join(data_dir, 'y_exp_val.npy'))
        y_gen_train = np.load(os.path.join(data_dir, 'y_gen_train.npy'))
        y_gen_val = np.load(os.path.join(data_dir, 'y_gen_val.npy'))
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

    # Create model
    try:
        model = create_mlp_model()
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return None

    # Define EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Combine callbacks
    callbacks = callbacks or []
    callbacks.append(early_stopping)

    # Train model
    try:
        history = model.fit(
            X_train,
            {'age_output': y_age_train, 'exp_output': y_exp_train, 'gen_output': y_gen_train},
            validation_data=(X_val, {'age_output': y_age_val, 'exp_output': y_exp_val, 'gen_output': y_gen_val}),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks
        )
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save(os.path.join(output_dir, 'model.h5'))
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return None

    # Save label encoders with detailed error handling
    encoder_files = ['le_age.pkl', 'le_expression.pkl', 'le_gender.pkl']
    for encoder_file in encoder_files:
        source_path = os.path.join(data_dir, encoder_file)
        dest_path = os.path.join(data_dir, encoder_file)
        if os.path.exists(source_path):
            try:
                with open(source_path, 'rb') as f_in:
                    encoder_data = pickle.load(f_in)
                with open(dest_path, 'wb') as f_out:
                    pickle.dump(encoder_data, f_out)
                print(f"Successfully saved {encoder_file} to {dest_path}")
            except EOFError as e:
                print(f"Warning: {encoder_file} is empty or corrupt: {str(e)}")
            except Exception as e:
                print(f"Error processing {encoder_file}: {str(e)}")
        else:
            print(f"Warning: {encoder_file} not found at {source_path}")

    print(f"Model saved to {output_dir}/model.h5")
    return history

if __name__ == "__main__":
    train_model('../Dataset/Training_Data', '../01_Mediapipe_Eksplorasi/Model')