import numpy as np
from pso import PSO

class PSOforANN:
    
    #Initialize class instance 
    def __init__(self, X, y, neural_network, swarm_size=20, beta=1.5, gamma=1.5, delta=0.5, alpha=0.5, epsilon=0.01,
                 informantType=0, informantNb=3, max_iters=100, print_iter=True):
        #Initialize class variables
        self.neural_network = neural_network 
        self.X = X
        self.y = y
        self.dimensions = len(neural_network.get_weights_and_biases())  #Calculate dimensionality for PSO based on number of weights and biases in neural network
        #Initialize PSO with passed values
        self.pso = PSO(func=self.fitness_function, swarm_size=swarm_size, dimensions=self.dimensions, 
                       beta=beta, gamma=gamma, delta=delta, alpha=alpha, epsilon=epsilon, 
                       informantType=informantType, informantNb=informantNb, max_iters=max_iters, print_iter=print_iter)
    #Function to calculate particle fitness, mean square error for neural networks 
    def fitness_function(self, weights):
        #Set weights in the neural network to particle position
        self.neural_network.set_weights_and_biases(weights)
        #Get predictions for training istances
        predictions = []
        for i in range(len(self.X)):
            output = self.neural_network.foward_propagation(self.X[i])
            predictions.append(output.item())
        #Compute mean squared error for prediction by neural network
        error = np.mean((np.array(predictions) - self.y) ** 2) 
        return error

    def optimize(self):
        #Run PSO to find the best weights for minimizing the fitness function
        best_weights = self.pso.evolve()
        #Set weights of neural network to those of best global position
        self.neural_network.set_weights_and_biases(best_weights)
        return best_weights  