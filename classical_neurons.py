#train_acc: 1.000
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

def preprocess(data):
    return data.reshape(-1, 784).astype(np.float32) / 255.0

X_train = preprocess(x_train_raw)
X_test = preprocess(x_test_raw)
y_train = y_train_raw.astype(np.int64)
y_test = y_test_raw.astype(np.int64)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class BipolarSTE(nn.Module):
    def forward(self, x):
        bipolar = torch.where(x >= 0, torch.tensor(1.0, device=x.device), torch.tensor(-1.0, device=x.device))
        return bipolar.detach() - torch.tanh(x).detach() + torch.tanh(x)


class ClassicalNeurons(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 40)
        self.activation = BipolarSTE()

    def forward(self, x):
        h = self.activation(self.fc1(x))
        o = self.activation(self.fc2(h))
        
        votes = o.view(-1, 10, 4).sum(dim=2) 
        return votes

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.002)


model = ClassicalNeurons()
trainer = pl.Trainer(max_epochs=13, limit_train_batches=1000, enable_progress_bar=True)

trainer.fit(model, train_loader)

model.eval()
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for x, y in test_loader:
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        y_pred_list.extend(preds.numpy())
        y_true_list.extend(y.numpy())

final_acc = accuracy_score(y_true_list, y_pred_list)

cm = confusion_matrix(y_true_list, y_pred_list)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Przewidziana Cyfra')
plt.ylabel('Prawdziwa Cyfra')
plt.title(f'Classic Neurons (Test Accuracy: {final_acc*100:.2f}%)', fontsize=16)

plt.savefig('wykresy_kwantowy/classical_neurons_confusion.png')
plt.show()