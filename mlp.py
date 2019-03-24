# Author: chenstarx @ GitHub
# A three-layer fully-connected MLP, trained by back propagation and gradient descent.
# Only requiring numpy

import numpy

class three_layer_perceptron :

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate) :

        # initiallize number of nodes of each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # learning rate of the MLP
        self.learning_rate = learning_rate

        # weight matrices. the initial weights are normal distributed.
        self.weights_input_hidden = numpy.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
        self.weights_hidden_output = numpy.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))

        # Below are built-in functions

        # sigmoid activation function
        self.activation = lambda x: 1 / (1 + numpy.exp(-x))
    
    # show the information of the object network
    def info(self) :
        print("Learning rate:", self.learning_rate)
        print("Input layer:", self.input_nodes, "nodes")
        print("Hidden layer:", self.input_nodes, "nodes")
        print("Output layer:", self.input_nodes, "nodes")
        print("Weights matrix of input and hidden layer:\n", self.weights_input_hidden)
        print("Weights matrix of hidden and output layer:\n", self.weights_hidden_output)

    # train the network
    def train(self, inputs_data, targets_data) :
        # transform the inputs and targets data set to matrix
        # inputs_data and targets_data should be 1-dimentional array
        inputs = numpy.array(inputs_data, ndmin=2).T
        targets = numpy.array(targets_data, ndmin=2).T


        # <-- forward computing -->

        # input matrix of hidden layer
        hidden_inputs = numpy.dot(self.weights_input_hidden, inputs)
        # output matrix of hidden layer
        hidden_outputs = self.activation(hidden_inputs)
        # input matrix of output layer
        final_inputs = numpy.dot(self.weights_hidden_output, hidden_outputs)
        # final out put matrix
        final_outputs = self.activation(final_inputs)
        # </-- forward computing -->


        # <-- backward computing -->

        # the errors matrix of output layer
        errors_output = targets - final_outputs
        # the errors matrix of hidden layer
        errors_hidden = numpy.dot(self.weights_hidden_output.T, errors_output)
        # </-- backward computing -->


        # <-- weights updating -->
        
        # the weights changes between hidden and output layer
        delta_weights_hidden_output = self.learning_rate * numpy.dot(errors_output * final_outputs * (1 - final_outputs), numpy.transpose(hidden_outputs)) #final_inputs

        # the weights changes between input and hidden layer
        delta_weights_input_hidden = self.learning_rate * numpy.dot(errors_hidden * hidden_outputs * (1 - hidden_outputs), numpy.transpose(inputs)) #hidden_inputs

        # update weights
        self.weights_hidden_output += delta_weights_hidden_output
        self.weights_input_hidden += delta_weights_input_hidden
        # </-- weights updating -->

    # query the networkw
    def query(self, inputs_data) :
        # transform inputs data set to a matrix
        inputs = numpy.array(inputs_data, ndmin=2).T

        # input matrix of hidden layer
        hidden_inputs = numpy.dot(self.weights_input_hidden, inputs)

        # output matrix of hidden layer
        hidden_outputs = self.activation(hidden_inputs)
        
        # input matrix of output layer
        final_inputs = numpy.dot(self.weights_hidden_output, hidden_outputs)

        # final out put matrix
        final_outputs = self.activation(final_inputs)

        return final_outputs

    def test_accuracy(self, test_file) :
        mnist_data = open(test_file)
        mnist_data_test = mnist_data.readlines()
        mnist_data.close()

        count = 0

        for image in mnist_data_test:

            image_pixels = image.split(',')

            inputs = (0.99 * numpy.asfarray(image_pixels[1:]) / 255.0) + 0.01

            number = image_pixels[0]

            result = self.query(inputs)

            prob = 0
            index = 0
            res_num = 0
            for res in result:
                value = res[0]
                if value > prob:
                    prob = value
                    res_num = index
                index += 1

            if int(res_num) == int(number):
                count += 1

        accuracy = 100 * count / len(mnist_data_test)

        print("The test accuracy is:", accuracy, "%")

    def load(self, weights_input_hidden_file, weights_hidden_output_file) :
        f = open(weights_input_hidden_file, "rb")
        self.weights_input_hidden = numpy.fromstring(f.read(), dtype=float)
        self.weights_input_hidden.shape = self.hidden_nodes, self.input_nodes
        f.close()

        f = open(weights_hidden_output_file, "rb")
        self.weights_hidden_output = numpy.fromstring(f.read(), dtype=float)
        self.weights_hidden_output.shape = self.output_nodes, self.hidden_nodes
        f.close()

    def save(self) :
        f = open("weights_hidden_output.bin", "wb")
        f.write(self.weights_hidden_output.tostring())
        f.close()

        f = open("weights_input_hidden.bin", "wb")
        f.write(self.weights_input_hidden.tostring())
        f.close()
# --- perceptron class end --- #


"""
# The codes below are used to train the network and get a weight map.

start = time.time()

mnist_data = open("mnist_train.csv")
mnist_data_list = mnist_data.readlines()
mnist_data.close()

perceptron = three_layer_perceptron(784, 300, 10, 0.003)

for i in range(0, 100):
    for image in mnist_data_list:

        image_pixels = image.split(',')

        inputs = (0.99 * numpy.asfarray(image_pixels[1:]) / 255.0) + 0.01

        targets = numpy.zeros(10) + 0.01
        targets[int(image_pixels[0])] = 0.99

        perceptron.train(inputs, targets)

perceptron.save() # save the weights to two txt files.

used = time.time() - start
print('Time used:', used)

perceptron.test_accuracy("mnist_test.csv")
"""
