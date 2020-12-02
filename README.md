# Introduction
The purpose of this projects is to show image classification with machine learning algorithm from scratch. That is why Pytorch or Tensorflow library is not used.
Softmax algorithm is used for Multinomial Logistic Regression to classify images.
Since the final accuracy was not satisfying enough adjusting learning rate and fitting more may increase it.

Cifar10 is used as dataset. It can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html

### Train
    $python train.py
      Loading weights...
      Training model...
      Epoch: 0
      Epoch: 500
      1000 epochs took 134.501 seconds.
      Saving weights...
      Final accuracy: 0.534
<p align="center">
          <img src="https://user-images.githubusercontent.com/45767042/100938503-d6377780-3505-11eb-93cb-24a6e6c464e1.png", width=640, height=480>
</p>

### Detection
    $python detection.py
<p align="center">
   <img src="https://user-images.githubusercontent.com/45767042/100941378-9626c380-350a-11eb-9be9-fcf708a94965.png", width=640, height=640>
</p>
