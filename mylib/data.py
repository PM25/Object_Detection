from pathlib import Path
import torch
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image


class TorchData():
    def __init__(self, base_path="data"):
        # Directory Path
        base_dir = Path("data")
        self.train_dir = base_dir / Path("train")
        self.test_dir = base_dir / Path("test")
        # Image transform Function
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
        ])


    # Load Training Data (x:features, y:labels)
    def load(self):
        x, y = [], []
        for file_name in self.train_dir.glob("*.jpg"):
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
            img = self.transform(img)
            x.append(img)

        return (x, y)


    def load_dataset(self):
        x, y = self.load()
        # Put training data into torch loader
        tensor_x = torch.stack(x)
        tensor_y = torch.stack(y)
        torch_dataset = Data.TensorDataset(tensor_x, tensor_y)

        return (torch_dataset)