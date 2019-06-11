import torch
import torchvision
from torch import nn
from torchvision import transforms
import torch.utils.data as Data
from pathlib import Path
import cv2
import numpy as np

models_dir = Path("models")
data_dir = Path("cats_and_dogs_small")
train_dir = data_dir/Path("train")
validation_dir = data_dir/Path("validation")
test_dir = data_dir/Path("test")

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
])


if __name__ == "__main__":
    model = torch.load(models_dir / "object_detection.pkl")
    model = model.cpu()

    cap = cv2.VideoCapture("dogs_and_cats.avi")

    while(cap.isOpened()):
        ret, frame = cap.read()
        torch_data = transform(cv2.resize(frame, (112, 112)))
        # torch_data = transform(frame)
        torch_data = torch.unsqueeze(torch_data, 0)
        (pos_x, pos_y, box_w, box_h) = model(torch_data)[0]
        height, width, channels = frame.shape
        pos_x *= width
        pos_y *= height
        box_w *= width
        box_h *= height
        img = cv2.rectangle(frame, (pos_x - box_w / 2, pos_y - box_h / 2), (pos_x + box_w / 2, pos_y + box_h / 2), (255, 0, 0), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
