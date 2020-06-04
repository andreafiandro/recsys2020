import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

pathtobig5scores = "../../dataset/myPersonalitySmall/big5labels.txt"
fb5 = pd.read_csv(pathtobig5scores, delim_whitespace=True, header=None)
dataset = pd.read_csv("csv_table.csv", header=None)
X = dataset  # myPersonality cls isa model
y = fb5.iloc[:, 4]  # working on Openness
y = y.to_numpy()
y = np.reshape(y, (len(y), 1))
inputs = torch.from_numpy(np.array(X))
targets = torch.from_numpy(y)
trains_ds = TensorDataset(inputs, targets)

num_epochs = 100
batch_size = 100
train_dl = DataLoader(trains_ds, batch_size=batch_size, shuffle=True)
model = nn.Sequential(nn.Linear(768, 300), nn.ReLU(), nn.Linear(300, 1))
# print(model.weight, model.bias)
loss_fn = F.mse_loss

# opt = torch.optim.SGD(model.parameters(), lr=1e-5)
opt = torch.optim.Adam(model.parameters(), lr=1e-5)  # best
# opt = torch.optim.Adagrad(model.parameters(), lr=1e-5) #worst
loss_array = []


def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb.float())
            loss = loss_fn(pred, yb.float())
            loss.backward()
            opt.step()
            opt.zero_grad()
        loss_array.append(loss.item())
        if(epoch+1) % 10 == 0:
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1,
                                                 num_epochs,
                                                 loss.item()))


fit(num_epochs, model, loss_fn, opt)
torch.save(model.state_dict(), "../models/SentPers_N")
