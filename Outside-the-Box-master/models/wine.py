import tensorflow.keras as kr
from tensorflow.keras import Input


def wines(classes, input_shape, weights=None):
    model = kr.models.Sequential()
    model.add(Input(shape=input_shape))
    model.add(kr.layers.Dense(13, activation='relu'))
    model.add(kr.layers.Dense(13, activation='relu'))
    model.add(kr.layers.Dense(13, activation='relu'))
    model.add(kr.layers.Dense(classes, activation='softmax'))

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return model
