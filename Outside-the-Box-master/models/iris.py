from tensorflow.keras.layers import Convolution2D
import tensorflow.keras as kr
from tensorflow.keras import Input


def iris():
    model = kr.models.Sequential()
    model.add(Input(shape=(4,)))
    # Add an initial layer with 4 input nodes, and a hidden layer with 16 nodes.
    model.add(kr.layers.Dense(16))
    # Apply the sigmoid activation function to that layer.
    model.add(kr.layers.Activation("sigmoid"))
    # Add another layer, connected to the layer with 16 nodes, containing three output nodes.
    model.add(kr.layers.Dense(3))
    # Use the softmax activation function there.
    model.add(kr.layers.Activation("softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


    return model
