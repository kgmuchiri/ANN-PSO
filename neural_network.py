import numpy as np
class NeuralNetwork:
    def __init__(self,input_nodes, hidden_layers,output_nodes, activation_functions):
        '''
        Initialises a Neural Network that is structured based on the input configuration
        Arguments:
            Input Nodes: Number of nodes in the input layer (int)
            Hidden Layers: An array where the number of elements is the number of hidden layers and each element represents
                            the number of nodes in that hidden layer e.g [3,4] where the first hidden layer containes 3 nodes and the second contains 4 nodes (int Array)
            Output Nodes: Number of nodes in the output layer (int)
            Activation functions: An array of string objects representing the activation function to be performed in each layer. (String Array)
        '''
        self.input_nodes = input_nodes
        self.hidden_layers = hidden_layers
        self.output_nodes = output_nodes
        self.activation_functions = activation_functions
        
        #Creates an array with all layers for random weight initialization
        self.layer_nodes = [input_nodes] + hidden_layers + [output_nodes]
        self.weights=[]
        self.biases=[]
        

        #Random weight intialization
        for i in range(len(self.layer_nodes) - 1):
            self.weights.append(np.random.rand(self.layer_nodes[i], self.layer_nodes[i + 1]))
            self.biases.append(np.random.rand(self.layer_nodes[i + 1]))
    
    #Activation functions
    def activate(self,x,function):
        if function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif function == "relu":
            return np.maximum(0, x)
        elif function == "tanh":
            return np.tanh(x)
        elif function == "linear":
            return x
        else:
            raise ValueError("Invalid activation function input")
    
    #Forward propagation
    def foward_propagation(self, input_data):
        a = input_data
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activate(z, self.activation_functions[i])

        return a
    
    #Get and Set methods for weights
    def get_weights_and_biases(self):
        parameters = []
        for w in self.weights:
            parameters.extend(w.flatten())
        for b in self.biases:
            parameters.extend(b.flatten())
        return np.array(parameters)
    
    def set_weights_and_biases(self, parameters):
        index = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            w_size = np.prod(w_shape)
            self.weights[i] = np.reshape(parameters[index:index + w_size], w_shape)
            index += w_size
            b_shape = self.biases[i].shape
            b_size = np.prod(b_shape)
            self.biases[i] = np.reshape(parameters[index:index + b_size], b_shape)
            index += b_size