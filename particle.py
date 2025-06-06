import numpy as np
import sys
import random

class Particle:
    #Initialize the particle with a fitness function and dimension
    def __init__(self, func, dimension):

        self._func = func  #The fitness function to be optimized
        self._velocity = None  #Initial velocity of the particle (set later)
        self._position = None  #Current position of the particle
        self._pbest_position = None  #Personal best position of the particle
        self._pbest_fitness = sys.float_info.max  #Initialize the personal best fitness to a large value
        self._informants = []  #List to store informant particles for social interaction
        self.initPosition(dimension)  #Initialize position and velocity

    #Initialize the position and velocity of the particle randomly
    def initPosition(self, dim):
        self._position = np.random.uniform(-3, 3, dim) #Random position within the range [-3, 3]
        self._pbest_position = np.copy(self._position) #Set the personal best position as the initial position
        self._velocity = np.random.uniform(-1, 1, dim) #Random velocity within the range [-1, 1]

    #Compute Euclidean distance between particles
    def distance(self, p):
        return np.linalg.norm(self._position - p._position)

    #Find the nearest `informantNb` particles in the swarm to set as informants
    def findNeighbours(self, particles, informantNb):
        neighbours = []
        for particle in particles:
            dist = self.distance(particle)  #Compute distance to each particle
            neighbours.append((particle, dist))  #Store particle with its distance
        neighbours.sort(key=lambda tup: tup[1])  #Sort by distance
        return [n[0] for n in neighbours[:informantNb]]  #Return the nearest informantNb particles

    #Set the informants based on the chosen type (random, nearest or mixed neighbors)
    def setInformants(self, informantType, informantNb, particles):
        if informantType == 0:  #Random informants
            self._informants = random.sample(particles, informantNb)
        elif informantType == 1:  #Nearest neighbors
            self._informants = self.findNeighbours(particles, informantNb)
        elif informantType == 2: #Mixed informants
            half_informants = informantNb // 2
            random_informants = random.sample(particles, half_informants)
            nearest_informants = self.findNeighbours(particles, informantNb - half_informants)
            self._informants = random_informants + nearest_informants

    #Update the velocity based on the particle's best, informants' best, and global best positions
    def update_velocity(self, lbest_position, gbest_position, alpha, beta, gamma, delta):
        r1, r2, r3 = np.random.rand(3)  # Random numbers
        #Update velocity using cognitive, social, and global components
        inertia_component = alpha * self._velocity # Inertia effect (previous velocity)
        cognitive_component = beta * r1 * (self._pbest_position - self._position) #Cognitive component (own best)
        social_component = gamma * r2 * (lbest_position - self._position) #Social component (informants' best)
        global_component = delta * r3 * (gbest_position - self._position) #Global component (global best)

        #Update velocity by summing all components
        self._velocity = inertia_component + cognitive_component + social_component + global_component

    #Move particle, update the position and apply boundary handling
    def move(self, epsilon, position_bounds=(-1, 1)):
        self._position += epsilon * self._velocity

    #Calculate fitness and update personal best if current position is better
    def evaluate_fitness(self):
        fitness = self._func(self._position)
        if fitness < self._pbest_fitness: #If current position has better fitness
            self._pbest_fitness = fitness #Update personal best fitness
            self._pbest_position = np.copy(self._position) #Update personal best position