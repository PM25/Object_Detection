import numpy as np
import math
import torch
from torch import nn
import torch.utils.data as Data
import torchvision
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from mylib.data import TorchData
from mylib.model import Model

# Parameters
EPOCH = 20
LR = 1e-3

# Load Data
train_data = TorchData("data")
torch_dataset = train_data.load_dataset()

val_count = math.floor(len(torch_dataset) * 0.1)
train_count = len(torch_dataset) - val_count
train_dataset, val_dataset = Data.random_split(torch_dataset, [train_count, val_count])

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, num_workers=0)


# Load model
model = torchvision.models.resnet18(pretrained=True)
fc_in_size = model.fc.in_features
model.fc = nn.Linear(fc_in_size, 4)
model = model.cuda()

loss_func = nn.SmoothL1Loss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=LR)

# Training
model = Model(model)
model.train(train_loader, loss_func, opt, 10)