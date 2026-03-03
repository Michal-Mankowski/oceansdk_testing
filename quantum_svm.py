from tensorflow.keras.datasets import mnist
import neal
import numpy as np
from scipy.spatial.distance import pdist, squareform
import dimod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def preprocess(image):
    return np.where(image[:, 6:22, 6:22].reshape(-1, 256) > 127, 1, -1)

X_train = preprocess(X_train)
X_test = preprocess(X_test)

batch_size = 1000
num_features = 256
num_classes = 10

sampler = neal.SimulatedAnnealingSampler()

K = 2
B = 2
lam = 10
gamma = 0.01

parameters = {}

for digit in range(num_classes):
    digit_idx = np.where(y_train == digit)[0]
    non_digit_idx = np.where(y_train != digit)[0]

    batch_idx = np.concatenate([
        np.random.choice(digit_idx, batch_size // 2),
        np.random.choice(non_digit_idx, batch_size // 2)
    ])

    x = X_train[batch_idx]
    y = np.where(y_train[batch_idx] == digit, 1 , -1)

    distances = squareform(pdist(x, 'sqeuclidean'))

    K_xn_xm = np.exp(-gamma * distances)

    
    Q_1 = (0.5 * K_xn_xm + lam) * np.outer(y, y)

    Q_2 = np.outer(B ** np.arange(K), B ** np.arange(K))
    Q = np.kron(Q_1, Q_2)

    diag_idx = np.arange(batch_size * K)
    Q[diag_idx, diag_idx] -= np.tile(B ** np.arange(K), batch_size)

    bqm = dimod.BinaryQuadraticModel(Q, "BINARY")

    response = sampler.sample(bqm, num_reads = 50)
    best_sample = response.first.sample

    sample_vec = np.array([best_sample[i] for i in range(len(best_sample))])
    sample_reshaped = sample_vec.reshape(batch_size, K)
    
    powers_of_B = B ** np.arange(K)
    alphas = np.dot(sample_reshaped, powers_of_B)
    
    support_indices = np.where(alphas > 0.01)[0]
    
    if len(support_indices) > 0:
        b_values = []
        for n in support_indices:
            k_n = np.exp(-gamma * np.sum((x - x[n])**2, axis=1))
            b_n = y[n] - np.sum(alphas * y * k_n)
            b_values.append(b_n)
        bias = np.mean(b_values)
    else:
        bias = 0

    parameters[digit] = {
        'alphas': alphas,
        'y': y,
        'x': x,
        'bias': bias
    }

def predict(X_new):
    scores = np.zeros((X_new.shape[0], num_classes))
    
    for digit, params in parameters.items():
        dists = pdist_to_train(X_new, params['x'])
        K_test = np.exp(-gamma * dists)
        
        scores[:, digit] = np.dot(K_test, params['alphas'] * params['y']) + params['bias']
    
    return np.argmax(scores, axis=1)

def pdist_to_train(A, B):
    sq_A = np.sum(A**2, axis=1).reshape(-1, 1)
    sq_B = np.sum(B**2, axis=1)
    return sq_A + sq_B - 2 * np.dot(A, B.T)


y_pred = predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'SVM (Test Accuracy: {accuracy*100:.2f}%)', fontsize=16)
plt.xlabel('Przewidziana Klasa')
plt.ylabel('Prawdziwa Klasa')
plt.savefig('wykresy_kwantowy/svm_confusion_matrix.png')
plt.show()
     

