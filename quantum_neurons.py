import numpy as np
import neal
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

def preprocess(data):
    return data.reshape(-1, 784) / 255.0

X_train = preprocess(x_train_raw)
X_test = preprocess(x_test_raw)

N_IN, N_HIDDEN, N_OUT = 784, 120, 40
W = np.random.uniform(-1/np.sqrt(N_IN), 1/np.sqrt(N_IN), (N_IN, N_HIDDEN))
J = np.random.uniform(-1/N_HIDDEN, 1/N_HIDDEN, (N_HIDDEN, N_OUT))
b_h = np.zeros(N_HIDDEN)
b_o = np.zeros(N_OUT)

sampler = neal.SimulatedAnnealingSampler()
lr = 0.002
m_samples = 10

def get_nudge(y):
    n = np.full(N_OUT, -1.0)
    n[4*y : 4*(y+1)] = 1.0
    return n


num_epochs = 13

for epoch in range(num_epochs):
    idx_batch = np.random.choice(len(X_train), 1000)
    hits = 0
    
    for idx in idx_batch:
        x, y = X_train[idx], y_train_raw[idx]
        nudge_vec = get_nudge(y)
        
        h_eff = np.dot(x, W) + b_h
        h_ising = {i: -h_eff[i] for i in range(N_HIDDEN)}
        for alpha in range(N_OUT):
            h_ising[N_HIDDEN + alpha] = -b_o[alpha]
            
        j_ising = {(i, N_HIDDEN + alpha): -J[i, alpha] 
                   for i in range(N_HIDDEN) for alpha in range(N_OUT)}
        
        res = sampler.sample_ising(h_ising, j_ising, num_reads=m_samples, sweeps=1000)
        
        samples = res.record.sample
        s_h_samples = samples[:, :N_HIDDEN]
        s_o_samples = samples[:, N_HIDDEN:]
        
        s_h_avg = np.mean(s_h_samples, axis=0)
        s_o_avg = np.mean(s_o_samples, axis=0)
        s_ho_corr_avg = (s_h_samples.T @ s_o_samples) / m_samples

        s_o_n = nudge_vec
        s_h_n = np.where(np.dot(x, W) + np.dot(s_o_n, J.T) + b_h >= 0, 1, -1)

        W += lr * np.outer(x, (s_h_n - s_h_avg))
        J += lr * (np.outer(s_h_n, s_o_n) - s_ho_corr_avg)
        b_h += lr * (s_h_n - s_h_avg)
        b_o += lr * (s_o_n - s_o_avg)

        votes = np.sum(s_o_avg.reshape(10, 4), axis=1)
        if np.argmax(votes) == y: hits += 1

    print(f"Epoch {epoch+1}/{num_epochs} - Accuracy: {hits/len(idx_batch)*100:.2f}%")

y_true = y_test_raw
y_pred = []

for i in range(len(y_true)):
    x = X_test[i]
    h_eff = np.dot(x, W) + b_h
    h_ising = {k: -h_eff[k] for k in range(N_HIDDEN)}
    for alpha in range(N_OUT): h_ising[N_HIDDEN + alpha] = -b_o[alpha]
    j_ising = {(k, N_HIDDEN + alpha): -J[k, alpha] for k in range(N_HIDDEN) for alpha in range(N_OUT)}
    
    res = sampler.sample_ising(h_ising, j_ising, num_reads=1, sweeps=1000)
    s_o = res.record.sample[0, N_HIDDEN:]
    y_pred.append(np.argmax(np.sum(s_o.reshape(10, 4), axis=1)))

final_acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Przewidziana Cyfra')
plt.ylabel('Prawdziwa Cyfra')

plt.title(f'Confusion Matrix \nValidation Accuracy: {final_acc*100:.2f}%')
plt.show()