import torch
from torch import nn
import torch.utils.data as Data
import torchvision


class Model:
    def __init__(self, model):
        self.model = model


    def train(self, loader, loss_func, optimizer, n_epochs=1, show_info=True):
        model = self.model.cuda()

        for epoch in range(n_epochs):
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

                output = model(batch_x)
                loss = loss_func(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (show_info and step % 5 == 0):
                    print("Epoch {} | Step {} | Loss {}".format(epoch, step, loss))


    def get_acc(self):
        pass