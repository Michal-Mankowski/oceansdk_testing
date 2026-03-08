# train accuracy: 91.79%

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns

@tf.custom_gradient
def binarize_op(x):
    result = tf.where(x >= 0, 1.0, -1.0)
    def grad(dy):
        return dy
    return result, grad

class BinaryDense(layers.Layer):
    def __init__(self, units, **kwargs):
        super(BinaryDense, self).__init__(**kwargs)
        self.units = units

    def build(self, output_shape):
        self.w = self.add_weight(
            shape=(output_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="weights"
        )

    def call(self, inputs):
        w_bin = binarize_op(self.w)
        return tf.matmul(inputs, w_bin)

def prepare_data():
    (x_full, y_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    indices = np.random.permutation(len(x_full))
    x_full = x_full[indices]
    y_full = y_full[indices]
    
    val_size = 10000
    x_val = x_full[-val_size:]
    y_val = y_full[-val_size:]
    x_train = x_full[:-val_size]
    y_train = y_full[:-val_size]

    def preprocess(x):
        x = x.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=-1)
        return x

    x_train = preprocess(x_train)
    x_val = preprocess(x_val)
    x_test = preprocess(x_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data()

conv_model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Resizing(24, 24, interpolation='bilinear'),
    
    layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=False),
    
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(2, 2))
])

weights_array = np.zeros((3, 3, 1, 4))
weights_array[:, :, 0, 0] = np.array([[-1, 0, 1],
                                        [-1, 0, 1],
                                        [-1, 0, 1]])
weights_array[:, :, 0, 1] = np.array([[-1, -1, -1],
                                        [0, 0, 0],
                                        [1, 1, 1]])
weights_array[:, :, 0, 2] = np.array([[ 0,  1,  1],
                                        [-1,  0,  1],
                                        [-1, -1,  0]])
weights_array[:, :, 0, 3] = np.array([[-1, -1,  0],
                                        [-1,  0,  1],
                                        [ 0,  1,  1]])

conv_model.set_weights([weights_array])
conv_model.trainable = False

model = models.Sequential([
    conv_model,
    layers.Flatten(),
    BinaryDense(10),
    layers.Softmax()
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Rozpoczynam trening...")
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=50, 
                    validation_data=(x_val, y_val),
                    verbose=1)

print("\nOstateczna ocena na zbiorze testowym:")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Dokładność testowa: {test_accuracy:.4f}")
print(f"Strata testowa: {test_loss:.4f}")

sample_imgs = x_test[:10]
sample_labels = y_test[:10]
predictions_vec = model.predict(sample_imgs, verbose=0)

plt.figure(figsize=(15, 6))
for i in range(10):
    prediction = np.argmax(predictions_vec[i])
    
    plt.subplot(2, 5, i + 1)
    display_img = sample_imgs[i].reshape(28, 28)
    plt.imshow(display_img, cmap='gray')
    color = 'green' if prediction == sample_labels[i] else 'red'
    plt.title(f"Pred: {prediction}\nTrue: {sample_labels[i]}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Classic CNN (Test Accuracy: {test_accuracy*100:.2f}%)', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('wykresy_kwantowy/classic_cnn_confusion_matrix.png')
plt.show()