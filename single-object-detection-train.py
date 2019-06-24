import matplotlib.pyplot as plt

from mylib.data import MyData
from mylib.model import Model


# Parameters
EPOCH = 20
log_interval = 1

# Record of Accuracy & Loss Value
acc_his = {"train": [], "val": []}
loss_his = {"train": [], "val": []}


if __name__ == "__main__":
    # Load Data
    train_data = MyData("data")
    train_loader, val_loader = train_data.load_split_dataloader(val_split=.2)

    model = Model(LR=1e-3, enable_cuda=True)
    for step in range(EPOCH):
        # Training
        model.train(train_loader, verbose=True)

        # Get Training Accuracy & Validation Accuracy
        train_acc = model.get_acc(train_loader, False)
        val_acc = model.get_acc(val_loader, False)
        acc_his["train"].append(train_acc)
        acc_his["val"].append(val_acc)

        if(step % log_interval == 0):
            print('-' * 15, end=' ')
            print("EPOCH {} | Train Acc {:.4f} | Val Acc {:.4f}".format(step, train_acc, val_acc), end=' ')
            print('-' * 15)

    # Show the accuracy result.
    for idx in acc_his:
        plt.plot(acc_his[idx], label=idx)
    plt.legend()
    plt.show()

    model.save_model("object_detection.pkl")