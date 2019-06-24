import torch
from torch import nn
import torchvision

from mylib import utils


class Model:
    def __init__(self, LR=1e-3, enable_cuda=True):
        self.enable_cuda = enable_cuda
        self.model = self.to_cuda(MyResnet(4, 2, enable_cuda))
        self.loss_func = self.to_cuda(nn.SmoothL1Loss())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.acc_func = utils.iou


    # Training the model
    def train(self, loader, n_epochs=1, show_info=True):
        self.model.train(True)
        self.model = self.to_cuda(self.model)

        for epoch in range(n_epochs):
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x = self.to_cuda(batch_x)
                batch_y = self.to_cuda(batch_y)

                box_preds, class_preds = self.model(batch_x)
                loss = self.loss_func(box_preds, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (show_info and step % 5 == 0):
                    print("Epoch {} | Step {} | Loss {}".format(epoch, step, loss))


    # Get Accuracy
    def get_acc(self, loader):
        self.model.eval()
        self.model = self.to_cuda(self.model)

        total_acc = 0
        total_count = 0
        for (batch_x, batch_y) in loader:
            batch_x = self.to_cuda(batch_x)
            box_preds, class_preds = self.model(batch_x)
            for (box_pred, y) in zip(box_preds, batch_y):
                acc = self.acc_func(y, box_pred)
                total_acc += acc
                total_count += 1

        avg_acc = total_acc/total_count
        return avg_acc


    def save_model(self, name="object_detection.pkl"):
        save_path = "models/" + name
        torch.save(self.model.cpu(), save_path)
        print("Model {} is saved!".format(name))


    def to_cuda(self, tensor):
        if(self.enable_cuda == True):
            tensor = tensor.cuda()

        return tensor


class MyResnet(nn.Module):
    def __init__(self, out1_sz, out2_sz, enable_cuda=True):
        super().__init__()
        if(enable_cuda == True):
            self.model_resnet = torchvision.models.resnet18(pretrained=True)
            self.model_resnet = self.model_resnet.cuda()
        else:
            self.model_resnet = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, out1_sz)
        self.fc2 = nn.Linear(num_ftrs, out2_sz)


    def forward(self, x):
        x = self.model_resnet(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2