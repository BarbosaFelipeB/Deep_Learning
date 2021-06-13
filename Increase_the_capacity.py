model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

# Wider networks have an easier time learning more linear relationships

wider = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Deeper networks prefer more nonlinear ones

deeper = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])