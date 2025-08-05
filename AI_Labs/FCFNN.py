import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

inputs = Input(shape=(3,))
hidden1 = Dense(4, activation='relu')(inputs)
hidden2 = Dense(4, activation='relu')(hidden1)
outputs = Dense(3, activation='softmax')(hidden2)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X = np.random.rand(10, 3)
y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
])

model.fit(X, y, epochs=100, batch_size=2)
predictions = model.predict(X)
print(predictions)
