import math
import torch.utils.data as Data
import matplotlib.pyplot as plt

from mylib.data import TorchData
from mylib.model import Model


# Parameters
EPOCH = 20


if __name__ == "__main__":
    # Load Data
    train_data = TorchData("data")
    torch_dataset = train_data.load_dataset()

    # Split Dataset to Training Dataset & Validation Dataset
    val_count = math.floor(len(torch_dataset) * 0.2)
    train_count = len(torch_dataset) - val_count
    train_dataset, val_dataset = Data.random_split(torch_dataset, [train_count, val_count])

    # Load Dataset into Torch DataLoader
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=True, num_workers=0)


    acc_his = { "train": [], "val":[] }
    loss_his = { "train": [], "val": [] }

    # Training
    model = Model()
    for step in range(EPOCH):
        print("Epoch {}".format(step))
        model.train(train_loader)
        acc_his["train"].append(model.get_acc(train_loader))
        acc_his["val"].append(model.get_acc(val_loader))

    model.save_model("test.pkl")
    for idx in acc_his:
        plt.plot(acc_his[idx], label=idx)
    plt.legend()
    plt.show()