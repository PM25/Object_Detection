from recognition import Recognition
import os
import matplotlib.pyplot as plt

base_dir = 'cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir=os.path.join(base_dir, 'test')


if __name__ == "__main__":
    model = Recognition(10)
    train_features, train_labels = model.feature_extraction(train_dir, 500)
    validation_features, validation_labels = model.feature_extraction(validation_dir, 250)
    test_features, test_labels = model.feature_extraction(test_dir, 250)
    history = model.fit_model(train_features, train_labels, validation_features, validation_labels)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training Acc')
    plt.plot(epochs, val_acc, 'b', label="Validation Acc")
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Traing and Validation Loss")
    plt.legend()
    plt.show()