import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

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

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
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
        x = (x.astype(np.float32) - 127.5) / 127.5
        x = x[..., tf.newaxis]
        x = tf.image.resize(x, [16, 16])
        x = tf.reshape(x, [-1, 256])
        return x

    x_train = preprocess(x_train)
    x_val = preprocess(x_val)
    x_test = preprocess(x_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data()

model = models.Sequential([
    layers.InputLayer(input_shape=(256,)),
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

trained_weights = model.layers[0].get_weights()[0] 
bw = np.where(trained_weights >= 0, 1, -1)
sample_imgs = x_test[:10].numpy()
sample_imgs_bin = np.where(sample_imgs >= 0, 1, -1)
sample_labels = y_test[:10]

plt.figure(figsize=(15, 6))
for i in range(10):
    img_vec = sample_imgs_bin[i]
    scores = np.dot(img_vec, bw)
    prediction = np.argmax(scores)
    plt.subplot(2, 5, i + 1)
    display_img = sample_imgs[i].reshape(16, 16)
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
cm = confusion_matrix(y_test, y_pred)

fig = plt.figure(figsize=(18, 8))

ax1 = plt.subplot(1, 3, 1)
ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
ax1.scatter(len(epochs_range), test_accuracy, color='green', s=150, 
            label=f'Test Accuracy: {test_accuracy:.4f}', zorder=5, marker='.', edgecolors='black')
ax1.set_title('Accuracy (Train, Val, Test)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([min(min(acc), min(val_acc), test_accuracy) - 0.05, 1.0])

ax2 = plt.subplot(1, 3, 2)
ax2.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
ax2.scatter(len(epochs_range), test_loss, color='green', s=150,
            label=f'Test Loss: {test_loss:.4f}', zorder=5, marker='.', edgecolors='black')
ax2.set_title('Loss (Train, Val, Test)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(1, 3, 3)
im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax3.set_title(f'Confusion Matrix\nTest Accuracy: {test_accuracy:.4f}', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

tick_marks = np.arange(10)
ax3.set_xticks(tick_marks)
ax3.set_yticks(tick_marks)
ax3.set_xticklabels(tick_marks)
ax3.set_yticklabels(tick_marks)
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax3.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10)

plt.tight_layout()
plt.show()