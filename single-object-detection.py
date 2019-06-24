import torch
from pathlib import Path
import argparse

from mylib.data import MyData
from mylib.predict import Predict

# Arguments
parser = argparse.ArgumentParser(description="Single Object Detection.")
parser.add_argument("--model", "-m", type=str, default="object_detection.pkl", help="Path to model.")
parser.add_argument("--image", "-i", type=str, default="img", help="Path to folder that contain image files.")
parser.add_argument("--video", "-v", type=str, default="video", help="Path to video file.")
parser.add_argument("--cuda", "-c", type=bool, default=False, help="Enable Cuda?")
args = parser.parse_args()

# Path
models_dir = Path("models")


# Start from here!
if __name__ == "__main__":
    model = torch.load(models_dir/args.model, map_location=lambda storage, loc: storage)
    if(args.cuda): model.cuda()
    else: model.cpu()

    names = MyData("data").names

    predict = Predict(model, names, img_dir=args.image, video_path=args.video)
    predict.img_prediction()
    predict.video_prediction()