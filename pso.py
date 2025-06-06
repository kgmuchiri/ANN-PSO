from particle import Particle
import numpy as np
class PSO:
    def __init__(self, func, dimensions, swarm_size=20, beta=1.5, gamma=1.5, delta=0.5, alpha=0.5, epsilon=0.01,
                 informantType=0, informantNb=3, max_iters=100, print_iter=True):
        self._func = func
        self._beta = beta  #Cognitive weight
        self._gamma = gamma  #Social weight
        self._delta = delta  #Global weight
        self._alpha = alpha  #Inertia weight
        self._epsilon = epsilon  #Step size
        self._particles = [] #List to hold all particles
        self._informantType = informantType #Type of informants (0 = random, 1 = nearest neighbors)
        self._informantNb = informantNb #Number of informants per particle
        self._max_iters = max_iters #Maximum number of iterations
        self._gbest_particle = None #Global best particle
        self.print_iter = print_iter  # Assign the print_iter parameter

        #Initialize the swarm with particles
        for i in range(swarm_size):
            particle = Particle(func, dimensions) #Create a new particle with the given function and dimensions
            self._particles.append(particle) #Add particle to swarm

        #Set informants for each particle
        for particle in self._particles:
            particles_copy = self._particles.copy()
            particles_copy.remove(particle)  #Remove current particle from informants selection
            particle.setInformants(informantType, informantNb, particles_copy) #Assign informants

        self.update_global_best() #Determine the initial global best
       

    def update_global_best(self):
        #Update global best particle in the swarm
        for particle in self._particles:
            #Update global best if current particle's best is better
            if self._gbest_particle is None or particle._pbest_fitness < self._gbest_particle._pbest_fitness:
                self._gbest_particle = particle 
                #print("Global best: ", self._gbest_particle._pbest_fitness)

    #Main loop to evolve the swarm
    def evolve(self):
        i = 0
        while i < self._max_iters:
            #Update the global best particle in the swarm
            self.update_global_best()

            #Iterate through particles to update their velocity and position
            for particle in self._particles:
                #Find the local best particle from informants
                lbest_particle = min(particle._informants, key=lambda p: p._pbest_fitness)

                #Update the particle's velocity based on the best positions and weights
                particle.update_velocity(lbest_particle._pbest_position, self._gbest_particle._pbest_position,
                                         self._alpha, self._beta, self._gamma, self._delta)
                particle.move(self._epsilon) #Move particle based on updated velocity
                particle.evaluate_fitness() #Evaluate fitness and possibly update personal best
            i += 1
            if ((i % (self._max_iters // 10) == 0) or i==1) and self.print_iter==True:  #Progress output every 10% of iterations
               print(f"Iteration {i}: Global Best Fitness = {self._gbest_particle._pbest_fitness}")
        
        return self._gbest_particle._pbest_position #Return the best position found

#Main Function with User Input
if __name__ == "__main__":

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
        dimensions = 2  #Example for a 2D optimization problem
    else:
        #Use default parameter values
        beta = 1.5
        gamma = 1.5
        delta = 0.5
        alpha = 0.5
        epsilon = 0.01
        population_size = 20
        max_iters = 100
        dimensions = 2  #Example for a 2D optimization problem
        print("\nRunning with default parameters:")
        print(f"beta = {beta}, gamma = {gamma}, delta = {delta}, alpha = {alpha}, epsilon = {epsilon}")
        print(f"population_size = {population_size}, max_iters = {max_iters}, dimensions = {dimensions}\n")

    #Define a sample fitness function (Sphere function for testing)
    def fitness_function(position):
        return np.sum(position ** 2)  #Example fitness function

    #Initialize and run the PSO algorithm with user-specified parameters    
    pso = PSO(func=fitness_function, dimensions=dimensions, swarm_size=population_size, beta=beta, gamma=gamma, delta=delta, alpha=alpha, epsilon=epsilon, max_iters=max_iters)
    best_position = pso.evolve()

    #Output the best position and fitness found by the swarm
    print("Global Best Position:", best_position)
    print("Global Best Fitness:", fitness_function(best_position))