from torchvision import transforms
from pathlib import Path
import cv2
import torch


class Video:
    def __init__(self, path, model):
        self.base_path = Path("data") / Path("test")
        self.path = self.base_path / path
        self.model = model


    def load_video(self):
        for video_name in self.path.glob("*.avi"):
            cap = cv2.VideoCapture(str(video_name))
            yield cap


    def draw_box(self, img, x, y, w, h, label):
        height, width, channels = img.shape
        x *= width
        y *= height
        w *= width
        h *= height
        img = cv2.rectangle(img, (x - w / 2, y - h / 2), (x + w / 2, y + h / 2), (255, 0, 0), 3)
        cv2.putText(img, label, (x - w / 2, y - h / 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        return img


    def show_results(self):
        for cap in self.load_video():
            while (cap.isOpened()):
                ret, frame = cap.read()
                torch_data = self.model.transform(cv2.resize(frame, (112, 112)))
                torch_data = torch.unsqueeze(torch_data, 0)
                box_pred, class_pred = self.model.model(torch_data)
                x, y, w, h = box_pred.squeeze()
                _, indices = torch.max(class_pred, 1)
                frame = self.draw_box(frame, x, y, w, h, self.model.names[indices])
                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()



class Image:
    def __init__(self, path, model):
        self.base_path = Path("data") / Path("test")
        self.path = self.base_path / path
        self.model = model


    def load_img(self):
        for img_name in self.path.glob("*.jpg"):
            img = cv2.imread(str(img_name))
            yield img


    def predict(self, img):
        torch_data = self.model.transform(cv2.resize(img, (112, 112)))
        torch_data = torch.unsqueeze(torch_data, 0)
        box_pred, class_pred = self.model.model(torch_data)

        return (box_pred, class_pred)


    def draw_box(self, img, x, y, w, h, label):
        height, width, channels = img.shape
        x *= width
        y *= height
        w *= width
        h *= height
        img = cv2.rectangle(img, (x - w/2, y - h/2), (x + w/2, y + h/2), (255, 0, 0), 3)
        cv2.putText(img, label, (x-w/2, y-h/2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        return img


    def show_results(self):
        for (idx, img) in enumerate(self.load_img()):
            box_pred, class_pred = self.predict(img)
            x, y, w, h = box_pred.squeeze()
            _, indices = torch.max(class_pred, 1)
            img = self.draw_box(img, x, y, w, h, self.model.names[indices])
            cv2.imshow("INDEX {}".format(idx), img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()



class Predict:
    def __init__(self, model, names, img_dir=None, video_path=None):
        self.model = model
        self.image = None
        self.video = None
        self.names = names
        if(img_dir != None):
            self.image = Image(img_dir, self)
        if(video_path != None):
            self.video = Video(video_path, self)

        # Torch Transform Function
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
        ])


    def img_prediction(self):
        if(self.image != None):
            self.image.show_results()


    def video_prediction(self):
        if(self.video != None):
            self.video.show_results()