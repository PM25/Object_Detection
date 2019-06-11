import torch
from mylib.image import Image

class Model:
    def __init__(self, model):
        self.model = model


    # Training the model
    def train(self, loader, loss_func, optimizer, n_epochs=1, show_info=True):
        self.model.train(True)
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


    # Get Accuracy
    def get_acc(self, loader, acc_func):
        self.model.eval()
        model = self.model.cuda()

        total_acc = 0
        total_count = 0
        for (batch_x, batch_y) in loader:
            batch_x = batch_x.cuda()
            preds = self.model(batch_x)
            for (pred, y) in zip(preds, batch_y):
                acc = acc_func(y, pred)
                total_acc += acc
                total_count += 1

        avg_acc = total_acc/total_count
        return avg_acc


    def save(self, name="object_detection.pkl"):
        save_path = "models/" + name
        torch.save(self.model, save_path)
        print("Model {} is saved!".format(name))
