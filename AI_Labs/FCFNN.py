import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

# Step 1: Input Layer (3 input features)
inputs = Input(shape=(3,))

# Step 2: Hidden Layers (each with 4 neurons)
hidden1 = Dense(4, activation='relu')(inputs)
hidden2 = Dense(4, activation='relu')(hidden1)

# Step 3: Output Layer (3 neurons for 3 classes with softmax)
outputs = Dense(3, activation='softmax')(hidden2)

# Step 4: Build Model
model = Model(inputs=inputs, outputs=outputs)

# Step 5: Compile Model (Categorical classification)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Dummy Data (10 samples, 3 features)
X = np.random.rand(10, 3)  # 10 rows, 3 input features

# Labels: 10 samples, 3 classes → one-hot encoded
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

# Step 7: Train the model
model.fit(X, y, epochs=100, batch_size=2)

# Step 8: Prediction
predictions = model.predict(X)
print(predictions)
model.summary(show_trainable=True)
