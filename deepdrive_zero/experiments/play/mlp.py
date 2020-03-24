import numpy as np

import torch

torch.manual_seed(1)

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act1 = torch.nn.Tanh()
        # self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        # self.act2 = torch.nn.Tanh()

    def forward(self, x):
        hidden = self.fc1(x)
        act1 = self.act1(hidden)
        # output = self.fc2(act1)
        # output = self.act2(output)
        return act1


# CREATE RANDOM DATA POINTS
from sklearn.datasets import make_blobs
def blob_label(y, label, loc): # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target


x_train_rand = np.random.rand(1, 2)
x_train_ones = np.ones(shape=(1, 2))
x_train_zeroes = np.zeros(shape=(1, 2))
y_train = np.ones(shape=(1, 10))

y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))
x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
y_test = torch.FloatTensor(blob_label(y_test, 1, [1,2,3]))

model = Feedforward(input_size=2, hidden_size=10)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
epoch = int(20e9)
for epoch in range(epoch):
    optimizer.zero_grad()
    if epoch < 3280:
        x_train = torch.FloatTensor(x_train_zeroes)
    else:
        x_train = torch.FloatTensor(x_train_rand)

    # Forward pass
    y_pred = model(x_train)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), y_train)

    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()