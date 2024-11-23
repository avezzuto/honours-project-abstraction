import tensorflow.keras as kr
from tensorflow.keras import Input


def dermatology(classes, input_shape, weights=None):
    model = kr.models.Sequential([
        kr.layers.Dense(128, input_dim=input_shape, activation='relu'),  # First hidden layer
        kr.layers.Dropout(0.3),  # Dropout for regularization
        kr.layers.Dense(64, activation='relu'),  # Second hidden layer
        kr.layers.Dropout(0.2),
        kr.layers.Dense(32, activation='relu'),  # Third hidden layer
        kr.layers.Dense(classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
