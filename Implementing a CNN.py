%%capture
%pip install torch_summary

import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.optim import SGD, Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# the shape of X_train is (batch: 2, Channel: 1, Height: 4, width: 4).
X_train = torch.tensor([ # Dim 0 (Batch) size 2
  [ # Dim 1 size 1, First Element, for example first image
    [ # Dim 2, Channel Height: 4 (4 rows) , gray Channel
      [1,2,3,4],[2,3,4,5],[5,6,7,8],[1,3,4,5] # Dim 3 size 4
    ]
  ],
  [ # Dim 1 size 1, Second image
    [ # Dim 2, Channel Height: 4  , gray Channel
      [-1,2,3,-4],[2,-3,4,5],[-5,6,-7,8],[-1,-3,-4,-5]
    ]
  ]
  ]).to(device).float()

X_train /= 8
y_train = torch.tensor([0,1]).to(device).float()

def get_model():
    model = nn.Sequential(
        nn.Conv2d(1,1, kernel_size=3), # there is 1 channel in the input and we are extracting 1 channel from the output post-convolution (that is, we have 1 filter with a size of 3 x 3) using the nn.Conv2d method.
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1,1),
        nn.Sigmoid()
      ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_func = nn.BCELoss()
    return model, optimizer, loss_func

def train_batch(x, y, model, optimizer, loss_func):
    model.train()
    prediction = model(x)
    prediction = prediction.squeeze(1) # remove extra dimension to be the same shape as the y target 
    loss = loss_func(prediction, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

class dataset(Dataset):
    def __init__(self, x ,y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
    
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    
    def __len__(self):
        return len(self.x)

def get_data():
    data = dataset(X_train, y_train)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    return dataloader

model, optimizer, loss_func = get_model()
dataloader = get_data() 
epoch_losses = []
for epoch in range(2000):
    batch_loses = []
    for index, batch in enumerate(iter(dataloader)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_func)
        batch_loses.append(batch_loss)
    epoch_loss = np.mean(batch_loses)
    epoch_losses.append(epoch_loss)

x_axis = range(2000)
plt.plot(x_axis, epoch_losses)
plt.grid(True)
plt.show()
