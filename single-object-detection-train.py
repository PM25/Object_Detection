from pathlib import Path
import numpy as np
import math
import torch
from torch import nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Parameters
EPOCH = 20
LR = 1e-3

# Directory Path
base_dir = Path("data")
train_dir = base_dir/Path("train")
test_dir = base_dir/Path("test")

# Image transform Function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
])

# Load Training Data (x:features, y:labels)
x, y = [], []
for file_name in train_dir.glob("*.jpg"):
    bounding_box_file = file_name.with_suffix('.txt')

    with open(bounding_box_file) as file:
        lines = file.readlines()
        if (len(lines) > 1):
            continue
        else:
            line = lines[0].strip('\n')
            (classes, cen_x, cen_y, box_w, box_h) = list(map(float, line.split(' ')))
            torch_data = torch.FloatTensor([cen_x, cen_y, box_w, box_h])
            y.append(torch_data)

    img = Image.open(str(file_name)).convert('RGB')
    img = transform(img)
    x.append(img)

# Put training data into torch loader
tensor_x = torch.stack(x)
tensor_y = torch.stack(y)
torch_dataset = Data.TensorDataset(tensor_x, tensor_y)

val_count = math.floor(len(torch_dataset) * 0.1)
train_count = len(torch_dataset) - val_count
train_dataset, val_dataset = Data.random_split(torch_dataset, [train_count, val_count])

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=1)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, num_workers=1)

# Load model
model = torchvision.models.resnet18(pretrained=True)
fc_in_size = model.fc.in_features
model.fc = nn.Linear(fc_in_size, 4)
model = model.cuda()

loss_func = nn.SmoothL1Loss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=LR)

# Training
his = {"train": [], "val": []}
loader = {"train": train_loader, "val": val_loader}
for epoch in range(EPOCH):
    model.train(True)

    for idx in loader:
        if (idx == "train"):
            model = model.cuda()
            model.train(True)
        else:
            model = model.cpu()
            model.train(False)

        total_loss = 0
        count = 0
        for step, (batch_x, batch_y) in enumerate(loader[idx]):
            if (idx == "train"):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            output = model(batch_x)
            loss = loss_func(output, batch_y)
            count += 1
            total_loss += loss.item()
            if (idx == "train"):
                opt.zero_grad()
                loss.backward()
                opt.step()

            if (step % 5 == 0):
                print("Epoch {} | Step {} | Loss {}".format(epoch, step, loss))
        his[idx].append(total_loss / count)
        print("Epoch {} | Loss {}".format(epoch, total_loss / count))