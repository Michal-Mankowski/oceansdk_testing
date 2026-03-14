import numpy as np
import neal
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from time import perf_counter_ns

print("Przygotowywanie danych...")
(x_all, y_all), (x_test_raw, y_test) = mnist.load_data()

num_features = 196

def preprocess(data):
    return np.where(data[:, 7:21, 7:21].reshape(-1, num_features) > 127, 1, -1)

x_train_full = preprocess(x_all)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_all, test_size=0.2, random_state=42)
x_test = preprocess(x_test_raw)

batch_size = 2000
alpha = 0.013
num_classes = 10

sampler = neal.SimulatedAnnealingSampler()
trained_weights = np.zeros((num_classes, num_features))
energies = []

times_classical = 0
times_quantum = 0

for digit in range(num_classes):
    
    start_time_classical = perf_counter_ns()
    pos_idx = np.where(y_train == digit)[0]
    neg_idx = np.where(y_train != digit)[0]
    
    batch_idx = np.concatenate([
        np.random.choice(pos_idx, batch_size // 2),
        np.random.choice(neg_idx, batch_size // 2)
    ])
    
    x_batch = x_train[batch_idx]
    y_target = np.where(y_train[batch_idx] == digit, 1, 0)
    
    J_batch = np.dot(x_batch.T, x_batch) * alpha
    h_batch = np.dot(x_batch.T, y_target)
    
    h_dict = {i: -2 * h_batch[i] for i in range(num_features)}
    J_dict = {(i, j): 2 * J_batch[i, j] for i in range(num_features) for j in range(i+1, num_features)}

    end_time_classical = perf_counter_ns()
    times_classical += end_time_classical - start_time_classical

    start_time_quantum = perf_counter_ns()
    sampleset = sampler.sample_ising(h_dict, J_dict, num_reads=10, sweeps=1000)
    end_time_quantum = perf_counter_ns()
    times_quantum += end_time_quantum - start_time_quantum

    best_sample = sampleset.first.sample
    trained_weights[digit] = np.array([best_sample[i] for i in range(num_features)])
    energies.append(sampleset.first.energy)


np.save('modele_symulowane/xnornet_14x14_weights.npy', trained_weights)
print(f"Classical time: {times_classical/1e9}")
print(f"Quantum simulated time: {times_quantum/1e9}")



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
#plt.savefig('wykresy_kwantowy/accuracy.png')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(range(10), energies, marker='s', markersize=8, color='blue', linewidth=2)
plt.xticks(range(10))
plt.xlabel('Cyfra')
plt.ylabel('Energia')
plt.title('Minimalizowana energia na klasę', fontsize=16)
plt.grid(True, alpha=0.2)
#plt.savefig('wykresy_kwantowy/energia.png')
plt.show()


plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'XNORNET (14x14) (Test Accuracy: {test_acc*100:.2f}%)', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('wykresy_kwantowy/xnornet14_confusion_matrix.png')
plt.show()
