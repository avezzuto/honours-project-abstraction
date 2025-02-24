import tensorflow.keras as kr
from tensorflow.keras import Input


def wines(classes, input_shape, weights=None):
    model = kr.models.Sequential()
    model.add(Input(shape=input_shape))

    model.add(kr.layers.Dense(13, use_bias=False))
    model.add(kr.layers.BatchNormalization())
    model.add(kr.layers.ReLU())

    # Second hidden layer with BatchNorm
    model.add(kr.layers.Dense(13, use_bias=False))
    model.add(kr.layers.BatchNormalization())
    model.add(kr.layers.ReLU())

    # Third hidden layer with BatchNorm
    model.add(kr.layers.Dense(13, use_bias=False))
    model.add(kr.layers.BatchNormalization())
    model.add(kr.layers.ReLU())

    model.add(kr.layers.Dense(classes, activation='softmax'))

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return model
