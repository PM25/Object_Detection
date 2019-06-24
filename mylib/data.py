import math
from pathlib import Path
import torch
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image


class MyData():
    def __init__(self, base_path="data"):
        # Directory Path
        self.base_dir = Path("data")
        self.train_dir = self.base_dir / Path("train")
        self.test_dir = self.base_dir / Path("test")
        self.names = self.load_names()
        self.classes_count = len(self.names)
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
                    torch_data = torch.FloatTensor([1] + [cen_x, cen_y, box_w, box_h] + self.one_hot_encode(classes))
                    y.append(torch_data)

            img = Image.open(str(file_name)).convert('RGB')
            img = self.transform(img)
            x.append(img)

        return (x, y)


    def load_names(self):
        names_file = self.base_dir/Path("names.txt")
        names = []
        with open(names_file) as file:
            for name in file.readlines():
                names.append(name)

        return names

    def load_dataset(self):
        x, y = self.load()
        # Put training data into torch loader
        tensor_x = torch.stack(x)
        tensor_y = torch.stack(y)
        dataset = Data.TensorDataset(tensor_x, tensor_y)

        return dataset


    def load_split_dataset(self, val_split=.2):
        # Split Dataset to Training Dataset & Validation Dataset
        dataset = self.load_dataset()
        val_count = math.floor(len(dataset) * val_split)
        train_count = len(dataset) - val_count
        split_dataset = Data.random_split(dataset, [train_count, val_count])

        return split_dataset


    def load_dataloader(self, batch_size=64, shuffle=True, num_workers=0):
        dataset = self.load_dataset()
        loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return loader


    def load_split_dataloader(self, val_split=.2, batch_size=64, shuffle=True, num_workers=0):
        train_dataset, val_dataset = self.load_split_dataset(val_split=val_split)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = Data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return (train_loader, val_loader)


    def one_hot_encode(self, idx):
        out = [0] * self.classes_count
        out[int(idx)] = 1

        return out