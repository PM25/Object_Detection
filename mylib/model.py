import torch
from torch import nn
import torchvision

from mylib import utils


class Model:
    def __init__(self, LR=5e-3, enable_cuda=True):
        self.enable_cuda = enable_cuda
        self.model = self.to_cuda(MyResnet(4, 2, enable_cuda))
        self.loss_func = self.to_cuda(nn.SmoothL1Loss())
        self.loss_func2 = self.to_cuda(nn.CrossEntropyLoss())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.acc_func = utils.iou


    # Training the model
    def train(self, loader, verbose=True):
        self.model.train(True)
        self.model = self.to_cuda(self.model)

        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x = self.to_cuda(batch_x)
            batch_y = self.to_cuda(batch_y)
            batch_box_y = batch_y[:, 1:5]
            batch_class_y = self.to_cuda(batch_y[:, -1].type(torch.LongTensor))

            box_preds, class_preds = self.model(batch_x)
            box_loss = self.loss_func(box_preds, batch_box_y)
            class_loss = self.loss_func2(class_preds, batch_class_y)
            loss = (box_loss + class_loss)/2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (verbose and step % 5 == 0):
                print("Step {} | Loss {:.4f}".format(step, loss.item()))


    # Get Accuracy
    def get_acc(self, loader, enable_cuda=True):
        self.model.eval()
        self.model = self.to_cuda(self.model, enable_cuda)

        total_acc = 0
        total_count = 0
        for (batch_x, batch_y) in loader:
            batch_x = self.to_cuda(batch_x, enable_cuda)
            batch_box_y = batch_y[:, 1:5]
            batch_class_y = self.to_cuda(batch_y[:, -1].type(torch.LongTensor))
            box_preds, class_preds = self.model(batch_x)
            # Accuracy of Bounding Box
            for (box_pred, y) in zip(box_preds, batch_box_y):
                acc = self.acc_func(y, box_pred)
                total_acc += acc
                total_count += 1
            # # Accuracy of Classification
            _, class_preds_indices = torch.max(class_preds, 1)
            class_preds_indices = class_preds_indices.cuda()
            total_acc += sum(class_preds_indices == batch_class_y).item()
            total_count += len(class_preds_indices)

        avg_acc = total_acc/total_count
        return avg_acc


    def save_model(self, name="object_detection.pkl"):
        save_path = "models/" + name
        torch.save(self.model.cpu(), save_path)
        print("Model {} is saved!".format(name))


    def to_cuda(self, tensor, enable=True):
        if(self.enable_cuda == True and enable == True):
            tensor = tensor.cuda()
        else:
            tensor = tensor.cpu()

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