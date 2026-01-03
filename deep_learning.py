import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# -----------------------------
# 2. Create Model
# -----------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# -----------------------------
# 3. Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 4. Train Model
# -----------------------------
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

# -----------------------------
# 6. Visualize Accuracy
# -----------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
