import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle


def create_training_data(file_path="data/batches", class_file="data/classes.txt", dataset="data/dataset.npy"):
    # Read data with pickle
    with open(class_file, "r") as c_file:
        Classes = c_file.read().split("\n")
    # Create one hot encoder list
    one_hot_encoder = list()
    # Adjust the list with respect to classes
    for i in range(len(Classes)):
        encode = [0 for k in range(len(Classes))]
        encode[i] = 1
        one_hot_encoder.append(encode)
    # Turn to numpy array
    one_hot_encoder = np.array(one_hot_encoder)
    # Get all batches from data/batches folder
    batch_list = glob.glob(file_path + "/*")
    # Create training_data that will contain inputs and outputs
    training_data = list()
    # For each batch read and append them to training_data
    for batch_path in batch_list:
        batch_path = batch_path.replace("\\","/")
        with open(batch_path, "rb") as f:
            data_dict = pickle.load(f, encoding="bytes")
        for i,label in enumerate(data_dict[b"labels"]):
            training_data.append([data_dict[b"data"][i], one_hot_encoder[label]])
    # Turn to numpy array
    training_data = np.array(training_data)
    # Save created dataset
    np.save(dataset, training_data)


# Works as sklearn Standard Scalar
def data_transformer(x):
    return (x - np.mean(x)) / np.std(x)


# Shows images and predictions in figure
def visualization(input_images=None, _classes=None):
    # Image container
    images = list()
    # Since for the every input red, green and blue  digits are from 0->1024, 1024->2048, 2048->3072
    # Obtained every one of them and turned into (32,32,3)
    # Reshaping from one lined array to (32,32,3) didn't work, so solved like below.
    # May need to think about more clear solution
    for k in range(input_images.shape[0]):
        # Make rgb list empty for every k
        image = list()
        # Obtain digits
        red = input_images[k][:1024].reshape(32, 32)
        green = input_images[k][1024:2048].reshape(32, 32)
        blue = input_images[k][2048:].reshape(32, 32)
        for i in range(32):
            # Make point list empty for every i
            point = list()
            for j in range(32):
                # Define points
                point.append([red[i][j], green[i][j], blue[i][j]])
            # Create image
            image.append(point)
        # Append to images with turning to uint8
        images.append(np.array(image, dtype=np.uint8))
    # Create figure holder
    axes = []
    # Define figure and size
    fig = plt.figure(figsize=(9, 12))
    for n in range(input_images.shape[0]):
        # Add figure to holder
        axes.append(fig.add_subplot(3, 3, n + 1))
        # Match image with prediction of it
        axes[-1].set_title(f"{_classes[n]}")
        plt.imshow(images[n])
    plt.show()


