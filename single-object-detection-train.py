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


def xywh_to_coords(x, y, w, h):
    x1 = x - (w/2)
    x2 = x + (w/2)
    y1 = y - (h/2)
    y2 = y + (h/2)

    return (x1, y1, x2, y2)


def overlap_area(rect1, rect2):
    r1_x1, r1_y1, r1_x2, r1_y2 = rect1
    r2_x1, r2_y1, r2_x2, r2_y2 = rect2
    w = min(r1_x2, r2_x2) - max(r1_x1, r2_x1)
    h = min(r1_y2, r2_y2) - max(r1_y1, r2_y1)

    if(w > 0 and h > 0): return (w*h)
    else: return -1


def area(rect):
    (x1, y1, x2, y2) = rect
    w = abs(x1-x2)
    h = abs(y1-y2)

    return (w*h)


def iou(truth, pred):
    pred = pred.tolist()
    truth = truth.tolist()
    pred_rect = xywh_to_coords(pred[0], pred[1], pred[2], pred[3])
    truth_rect = xywh_to_coords(truth[0], truth[1], truth[2], truth[3])

    overlap_sz = overlap_area(pred_rect, truth_rect)
    total_sz = area(pred_rect) + area(truth_rect) - overlap_sz

    if(overlap_sz > 0): return (overlap_sz/total_sz)
    else: return -1


# Training
model = Model(model)
for i in range(10):
    model.train(train_loader, loss_func, opt, 1)
    # print(model.get_acc(train_loader, loss_func))
    print(model.get_acc(val_loader, iou))