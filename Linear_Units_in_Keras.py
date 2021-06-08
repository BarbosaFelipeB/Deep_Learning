from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit (output) and 3 features (inputs)
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])

# Get random weights and 0.0 bias with object.attribute

w, b = model.weights

print("Weights\n{}\n\nBias\n{}".format(w, b))

