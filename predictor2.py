import argparse
import time
import numpy as np
import yaml
import itertools
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from preprocessing_utilities import (
    read_img_from_path,
    resize_img,
    read_from_file,
)
#from utils import download_model


class ImagePredictor:
    def __init__(
        self, model_path, resize_size, targets, pre_processing_function=preprocess_input
    ):
        self.model_path = model_path
        self.pre_processing_function = pre_processing_function
        self.model = load_model(self.model_path)
        self.resize_size = resize_size
        self.targets = targets

    @classmethod
    def init_from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        predictor = cls(
            model_path=config["model_path"],
            resize_size=config["resize_shape"],
            targets=config["targets"],
        )
        return predictor

    @classmethod
    def init_from_config_url(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)

        download_model(
            config["model_url"], config["model_path"], config["model_sha256"]
        )

        return cls.init_from_config_path(config_path)

    def predict_from_array(self, arr):
        arr = resize_img(arr, h=self.resize_size[0], w=self.resize_size[1])
        arr = self.pre_processing_function(arr)
        pred = self.model.predict(arr[np.newaxis, ...]).ravel().tolist()
        pred = [round(x, 3) for x in pred]
        pred = {k: v for k, v in zip(self.targets, pred)}
        pred = {k: v for k, v in sorted(pred.items(), key=lambda item: item[1], reverse=True)}
        out = dict(itertools.islice(pred.items(), 3)) 
        names, percentages = zip(*out.items())
        #pred = sorted(pred, reverse=True)
        #pred = [pred[i]for i in range(3)]
        #namesStr = "what"
        return pred #names, percentages
        #return {k: v for k, v in zip(self.targets, pred)}

    def predict_from_path(self, path):
        """
        if platform.system() == "Windows":
            dash = "\\"
        else:
            dash = "/"
        # Establish main path for files # <-- Final GUI change
        path = dash.join(os.path.realpath(__file__).split(dash)[:-1]) + dash
        """
        arr = read_img_from_path(path)
        return self.predict_from_array(arr)

    def predict_from_file(self, file_object):
        arr = read_from_file(file_object)
        return self.predict_from_array(arr)


if __name__ == "__main__":
    """
    python predictor.py --predictor_config "../example/predictor_config.yaml"

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor_config_path",
        help="predictor_config_path",
        default="predictor_config.yaml",
    )

    args = parser.parse_args()

    predictor_config_path = args.predictor_config_path
    predictor_config_path = "predictor_config.yaml"
    predictor = ImagePredictor.init_from_config_path(predictor_config_path)
    
    t0 = time.time()

    pred = predictor.predict_from_path(
         "class_A_11.png"
     )
    print(pred)
    
    t1 = time.time()
    print(t1-t0)
    
    print("time for one class", t1-t0)
    t2=time.time()
    pred = predictor.predict_from_path(
         "class_A_11.png"
     )
    t3=time.time()
    print(pred)
    print("time:", t3-t2)
    
    with open("class_A_11.png", "rb") as f:
         _pred = predictor.predict_from_file(f)
    
    print(_pred)
    
