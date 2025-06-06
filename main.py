import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import itertools
import sys
import random
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from pso import PSO
from particle import Particle
from psoforann import PSOforANN



df = pd.read_csv('concrete_data.csv')
print('Dataset retrieved successfully')


scaler = MinMaxScaler()
#Applies the scaler to the features, excluding the target class
df_nrml = df.copy()
df_nrml[df.columns.difference(['concrete_compressive_strength'])] = scaler.fit_transform(df[df.columns.difference(['concrete_compressive_strength'])])
#Displays the normalized data to verify
df_nrml.head()
#Create features and labels
X = df_nrml.drop('concrete_compressive_strength', axis=1).valuesadd
y = df_nrml['concrete_compressive_strength'].values
#Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Dataset processed successfully')


if __name__ == "__main__":
    print("input and output nodes are determined from the number of features and labels 8 input nodes one output node \n")
    num_layers = int(input("Enter the number of hidden layers: "))
    hidden_layers = []
    activation_functions = []
    for i in range(num_layers):
        nodes = int(input(f"Enter the number of nodes in hidden layer {i+1}: "))
        hidden_layers.append(nodes)
        activation = input(f"Enter activation function for hidden layer {i+1} (sigmoid, relu, tanh, linear): ")
        activation_functions.append(activation)
    output_activation = input("Enter activation function for output layer (sigmoid, relu, tanh, linear): ")
    activation_functions.append(output_activation)

    #Nonmutable hyperparameters
    input_nodes = X.shape[1]
    output_nodes = 1

    ann = NeuralNetwork(input_nodes, hidden_layers, output_nodes, activation_functions)

        #Ask the user if they want to enter custom parameters
    use_custom_params = input("Do you want to enter custom parameters? (yes/no): ").strip().lower()
    
    if use_custom_params == "yes":
        #Prompt the user for PSO parameters
        beta = float(input("Enter cognitive weight (beta): "))
        gamma = float(input("Enter social weight (gamma): "))
        delta = float(input("Enter global weight (delta): "))
        alpha = float(input("Enter inertia weight (alpha): "))
        epsilon = float(input("Enter position update step size (epsilon): "))
        population_size = int(input("Enter swarm size (population size): "))
        max_iters = int(input("Enter maximum iterations: "))
        swarm_size = int(input("Enter swarm size (swarm size): "))
        informant_type = int(input("Enter type of informants (0 = random, 1 = nearest neighbors): "))
        informant_nb = int(input("Enter number of informants per particle: "))
        dimensions = 3
    else:
        #Use default parameter values
        beta = 1.5
        gamma = 1.5
        delta = 0.5
        alpha = 0.5
        epsilon = 0.01
        population_size = 20
        max_iters = 400
        dimensions = 3  
        informant_type = 0
        informant_nb = 3
        swarm_size = 50
        print("\nRunning with default parameters:")
        print(f"beta = {beta}, gamma = {gamma}, delta = {delta}, alpha = {alpha}, epsilon = {epsilon}")
        print(f"population_size = {population_size}, max_iters = {max_iters}, dimensions = {dimensions}\n")

    #Define a sample fitness function (Sphere function for testing)
    def fitness_function(position):
        return np.sum(position ** 2)  #Example fitness function

    #Initialize and run the PSO algorithm with user-specified parameters    
    pso = PSO(func=fitness_function, dimensions=dimensions, swarm_size=population_size, beta=beta, gamma=gamma, delta=delta, alpha=alpha, epsilon=epsilon, max_iters=max_iters)
    best_position = pso.evolve()



    #Initialize the neural network with set values
    ann = NeuralNetwork(input_nodes=input_nodes, hidden_layers=hidden_layers,
                        output_nodes=output_nodes, activation_functions=activation_functions)

    #Initialize the PSOforANN instance with the neural network and PSO parameters
    annOptimizer = PSOforANN(X=X_train, y=y_train, neural_network=ann, 
                        swarm_size=swarm_size, beta=beta, gamma=gamma, delta=delta, 
                        alpha=alpha, epsilon=epsilon, informantType=informant_type, 
                        informantNb=informant_nb, max_iters=max_iters)

    #Optimize the neural network using PSO
    best_weights = annOptimizer.optimize()

    #Set weights of ann to the best weights calculated
    ann.set_weights_and_biases(best_weights)

    #Function to calculate predictions
    def calc_error(neural_network, X_test, y_test):
        predictions = []
        for i in range(len(X_test)):
            output = neural_network.foward_propagation(X_test[i])  #Perform forward propagation
            predictions.append(output.item())  #Append prediction for instance to predictions list
        predictions
        #Calculate MSE on the test set
        mse = np.mean((np.array(predictions) - y_test) ** 2)
        return mse

    mse = calc_error(ann, X_test, y_test)
    print(f"Mean Squared Error (MSE) on test set with optimized weights: {mse}")