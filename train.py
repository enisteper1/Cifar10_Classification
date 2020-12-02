import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import create_training_data, data_transformer
from MultinomialLR_with_equations import MultinomialLogisticRegression
import time
from pathlib import Path


def train_model(args):
    # Initialize Parameters
    batch_folder, class_file, dataset, lr, epochs, every, weight_dir = args.batch_folder, args.class_file,\
                                                          args.dataset, args.lr, args.epochs,\
                                                                 args.every, args.weight_dir
    # Create dataset.npy (training data)
    create_training_data(class_file=class_file,file_path=batch_folder, dataset=dataset)
    # Define model
    mult_log_reg = MultinomialLogisticRegression()
    # Check if the weight exist
    if Path(weight_dir).exists():
        print("Loading weights...")
        weights = np.load(weight_dir, allow_pickle=True)
        mult_log_reg.theta_1, mult_log_reg.theta_2 = weights[0], weights[1]
    # Load dataset.npy
    loaded_dataset = np.load(dataset, allow_pickle=True)
    # Squeeze input between 0 and 1
    x = np.array([data[0] for data in loaded_dataset]) / 255.0
    y = np.array([data[1] for data in loaded_dataset])
    # Because of exponential of sum of input was overflowing input is divided by 100 again
    # Dividing by 100 may be the another reason of low accuracy
    x = data_transformer(x) / 100
    print("Training model...")
    start_time = time.time()
    # fitting returns accuracy progress
    history = mult_log_reg.fit(x, y, iteration_num=epochs, learning_rate=lr, every=every)
    history_str = "\n".join(history.astype("str").tolist())
    # Save the history
    with open("progress.txt", "a") as f:
        f.write("\n" + history_str)

    training_time = time.time() - start_time
    print(f"{epochs} epochs took {round(training_time,4)} seconds.")
    weights = [mult_log_reg.theta_1, mult_log_reg.theta_2]

    print("Saving weights...")
    np.save(weight_dir, weights)
    # Prediction
    pred = mult_log_reg.prediction(x)
    acc = mult_log_reg.accuracy(pred, y)
    print(f"Final accuracy: {round(acc, 4)}")
    # Read all progress from progress.txt
    with open("progress.txt", "r") as f:
        history_str = f.read().split("\n")
    # Turn all strings to float
    history = [float(each_hist) for each_hist in history_str]
    # Interval is used to plot accuracy history step by step
    interval = [num for num in range(len(history))]
    plt.plot(interval, history, color="red", label="Accuracy")
    plt.grid()
    plt.legend(loc=2)
    plt.savefig("Result.png")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_folder", type=str, default="data/batches",help="Data file")
    parser.add_argument("--class_file", type=str, default="data/classes.txt", help="File that contains your classes")
    parser.add_argument("--dataset", type=str, default="data/dataset.npy", help="Dataset name that will be created")
    parser.add_argument("--lr", type=float, default=10, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=1000, help="How many times the model will be trained")
    parser.add_argument("--every", type=int, default=50, help="Calculate accuracy for every selected number")
    parser.add_argument("--weight_dir", type=str, default="weights/model.npy")
    args = parser.parse_args()

    train_model(args)
