# MLP demo

This project realized a three-layer perceptron to recognize handwriting digit.

- It uses python to build and train a MLP.
- The MLP was built with basic codes, where `numpy` is the only used library.
- MNIST was used to train the network.
- Back propagation and gradient descent algorithms were implemented.

In order to better demonstrate the demo, a simple user interface was built.

- A front-end web interface was built to collect user's input.
- A python http server was implemented to handle the web request. It inputs the user's handwriting image in base64 and output the recognition result to front-end.

## project guide

- All the codes for MLP are in `mlp.py`. It contains functions for the MLP's inference and training.
- Back propagation and Gradient descent algorithms were implemented in the MLP's training function.
- `server.py` is only for the demo, which does not contain any code related to the neural network. 
- The MLP was already trained, the weights are saved in two .bin files of the project. The weights will be loaded when the server start running.

## How to run a demo

1. run server.py
2. open demo.html **through localhost**
3. plot a number from 0 to 9 on the page
4. waiting for the recognition result

## Training Hint

If you want to train the MLP by yourself, please go the link below and download the .csv training and testing data set `mnist_train.csv` and `mnist_test.csv`.

https://pjreddie.com/projects/mnist-in-csv/