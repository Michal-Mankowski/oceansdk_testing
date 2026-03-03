import numpy as np
import neal
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Resizing, Input, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

(x_all, y_all), (x_test, y_test) = mnist.load_data()

def cnn():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Resizing(24, 24, interpolation='bilinear'),
        
        Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=False),
        
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)) 
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
    
    model.set_weights([weights_array])
    model.trainable = False
    return model

cnn_model = cnn()

def preprocess_with_cnn(data, model):
    data_norm = data.astype('float32') / 255.0
    data_norm = np.expand_dims(data_norm, axis=-1)
    
    features = model.predict(data_norm, verbose=0)
    
    features_flat = features.reshape(features.shape[0], -1)
    
    return features_flat

x_all = preprocess_with_cnn(x_all, cnn_model)
x_test = preprocess_with_cnn(x_test, cnn_model)

x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

num_features = x_train.shape[1]

batch_size = 2000
alpha = 0.013
num_classes = 10

sampler = neal.SimulatedAnnealingSampler()
trained_weights = np.zeros((num_classes, num_features))
energies = []

for digit in range(num_classes):
    print(f"Trenowanie cyfry {digit}/9...")
    
    pos_idx = np.where(y_train == digit)[0]
    neg_idx = np.where(y_train != digit)[0]
    
    if len(pos_idx) < batch_size // 2:
        curr_pos = np.random.choice(pos_idx, batch_size // 2, replace=True)
    else:
        curr_pos = np.random.choice(pos_idx, batch_size // 2, replace=False)
        
    batch_idx = np.concatenate([curr_pos, np.random.choice(neg_idx, batch_size // 2)])
    
    x_batch = x_train[batch_idx]
    y_target = np.where(y_train[batch_idx] == digit, 1, 0)
    
    J_batch = np.dot(x_batch.T, x_batch) * alpha
    h_batch = np.dot(x_batch.T, y_target)
    
    h_dict = {i: -2 * h_batch[i] for i in range(num_features)}
    
    J_dict = {}
    for i in range(num_features):
        for j in range(i+1, num_features):
            if abs(J_batch[i, j]) > 0.08:
                J_dict[(i, j)] = 2 * J_batch[i, j]

    sampleset = sampler.sample_ising(h_dict, J_dict, num_reads=10, sweeps=1000)
    
    best_sample = sampleset.first.sample
    trained_weights[digit] = np.array([best_sample[i] for i in range(num_features)])
    energies.append(sampleset.first.energy)


def get_predictions(x):
    scores = np.dot(x, trained_weights.T)
    return np.argmax(scores, axis=1)

train_pred = get_predictions(x_train)
val_pred = get_predictions(x_val)
test_pred = get_predictions(x_test)

train_acc = np.mean(train_pred == y_train)
val_acc = np.mean(val_pred == y_val)
test_acc = np.mean(test_pred == y_test)


plt.figure(figsize=(8, 6))
labels = ['Train', 'Val', 'Test']
values = [train_acc * 100, val_acc * 100, test_acc * 100]
plt.bar(labels, values, color=['blue', 'red', 'green'], alpha=0.7)
plt.ylim(0, 100)
plt.title('Dokładność Modelu', fontsize=16)
plt.ylabel('Accuracy [%]')
for i, v in enumerate(values):
    plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')
plt.savefig('wykresy_kwantowy/cnn_accuracy.png')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(range(10), energies, marker='s', markersize=8, color='blue', linewidth=2)
plt.xticks(range(10))
plt.xlabel('Cyfra')
plt.ylabel('Energia')
plt.title('Minimalizowana energia na klasę', fontsize=16)
plt.grid(True, alpha=0.2)
plt.savefig('wykresy_kwantowy/cnn_energia.png')
plt.show()


plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'CNN (Test Accuracy: {test_acc*100:.2f}%)', fontsize=16)
plt.xlabel('Przewidziana Klasa')
plt.ylabel('Prawdziwa Klasa')
plt.savefig('wykresy_kwantowy/cnn_confusion_matrix.png')
plt.show()