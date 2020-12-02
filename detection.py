from MultinomialLR_with_equations import MultinomialLogisticRegression
from utils import data_transformer, visualization
from argparse import ArgumentParser
import numpy as np
import random
from pathlib import Path


def detect(dataset="data/dataset.npy", weight_dir="weights/model.npy", class_file="data/classes.txt"):
    with open(class_file, "r") as f:
        labels = f.read().split("\n")
    loaded_dataset = np.load(dataset, allow_pickle=True)
    # _x will be used for showing image and x for prediction input
    _x = np.array([data[0] for data in loaded_dataset])
    # Applying same steps again for input
    x = _x / 255.0
    x = data_transformer(x) / 100.0
    # Get random 9 image from batch data
    rand_num = random.randint(0, x.shape[0] - 9)
    # Input for prediction
    input_data = x[rand_num : rand_num + 9]
    # Image for visualization
    input_image = _x[rand_num:rand_num + 9]
    mult_log_reg = MultinomialLogisticRegression()
    # Check if weights exist
    if Path(weight_dir).exists():
        weights = np.load(weight_dir, allow_pickle=True)
        # Load weights
        mult_log_reg.theta_1, mult_log_reg.theta_2 = weights[0], weights[1]
    # Prediction
    predictions = mult_log_reg.prediction(input_data)
    # Match all one hot encoders with classes
    arg_pred_list = [labels[np.argmax(prediction)] for prediction in predictions]
    # Show Image and predictions
    visualization(input_images=input_image, _classes=arg_pred_list)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--class_file", type=str, default="data/classes.txt", help="File that contains your classes")
    parser.add_argument("--dataset", type=str, default="data/dataset.npy", help="Dataset name that will be created")
    parser.add_argument("--weight_dir", type=str, default="weights/model.npy")
    args = parser.parse_args()
    detect(args.dataset, args.weight_dir, args.class_file)