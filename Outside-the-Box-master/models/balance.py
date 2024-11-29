import tensorflow.keras as kr
from tensorflow.keras import Input


def balance(classes, input_shape, weights=None):
    model = kr.models.Sequential([
        kr.layers.Dense(16, input_dim=input_shape, activation='relu'),
        kr.layers.Dense(8, activation='relu'),
        kr.layers.Dense(classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
