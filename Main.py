import pickle
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


x = pickle.load(open(os.path.join(os.getcwd(), "features.pickle"), "rb"))
y = pickle.load(open(os.path.join(os.getcwd(), "labels.pickle"), "rb"))

x = x/255.0
x = np.float16(x)



model = keras.Sequential(
    [
        layers.Conv2D(64, (3,3), input_shape=x.shape[1:]),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2,2)),

        layers.Conv2D(64, (3,3)),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2,2)),

        layers.Conv2D(64, (3,3)),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2,2)),

        layers.Dropout(0.1),

        layers.Flatten(),
        
        layers.Dense(512, activation = "relu"),
        layers.Dense(512, activation = "relu"),
        layers.Dense(256, activation = "relu"),
        layers.Dense(128, activation = "relu"),
        layers.Dense(31, activation = "softmax")
    ]
)

model.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"],
)

model.fit(x, y, batch_size=64, epochs = 3, validation_split = 0.3)
