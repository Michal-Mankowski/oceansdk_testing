from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def preprocess(image):
    return np.where(image[:, 6:22, 6:22].reshape(-1, 256) > 127, 1, -1)

X_train = preprocess(X_train)
X_test = preprocess(X_test)

batch_size = 10000
indices = np.random.choice(len(X_train), batch_size, replace=False)
X_train_sub = X_train[indices]
y_train_sub = y_train[indices]

svm = SVC(kernel='rbf', gamma=0.01, C=1.0, decision_function_shape='ovr')

svm.fit(X_train_sub, y_train_sub)

y_pred = svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Classic SVM (Test Accuracy: {accuracy*100:.2f}%)', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('True')

plt.savefig('wykresy_kwantowy/classical_svm_confusion_matrix.png')
plt.show()