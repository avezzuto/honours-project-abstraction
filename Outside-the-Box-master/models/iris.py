from tensorflow.keras.layers import Convolution2D
import tensorflow.keras as kr
from tensorflow.keras import Input


def iris(classes, input_shape, weights=None):
    model = kr.models.Sequential()
    model.add(Input(shape=input_shape))
    # Add an initial layer with 4 input nodes, and a hidden layer with 16 nodes.
    model.add(kr.layers.Dense(16))
    # Apply the sigmoid activation function to that layer.
    model.add(kr.layers.Activation("sigmoid"))
    # Add another layer, connected to the layer with 16 nodes, containing three output nodes.
    model.add(kr.layers.Dense(3))
    # Use the softmax activation function there.
    model.add(kr.layers.Dense(classes, "softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
