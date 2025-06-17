import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def create_mlp_model(input_shape=(1404,), num_classes_age=3, num_classes_exp=4, num_classes_gen=2):
    """Create a multi-output MLP model for age, expression, and gender classification."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)

    age_output = layers.Dense(num_classes_age, activation='softmax', name='age_output')(x)
    exp_output = layers.Dense(num_classes_exp, activation='softmax', name='exp_output')(x)
    gen_output = layers.Dense(num_classes_gen, activation='softmax', name='gen_output')(x)

    model = Model(inputs=inputs, outputs=[age_output, exp_output, gen_output])

    model.compile(
        optimizer='adam',
        loss={
            'age_output': 'sparse_categorical_crossentropy',
            'exp_output': 'sparse_categorical_crossentropy',
            'gen_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'age_output': 'accuracy',
            'exp_output': 'accuracy',
            'gen_output': 'accuracy'
        }
    )

    return model

if __name__ == "__main__":
    model = create_mlp_model()
    model.summary()