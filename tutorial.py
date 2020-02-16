import mlrose_reborn as mlrose
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Tutorial from MLRose Documentation
# https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb
# 
# Initialize fitness function object using pre-defined class
fitness = mlrose.Queens()

# Define optimization problem object
problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=False, max_val=8)

# Define decay schedule
schedule = mlrose.ExpDecay()

# Solve using simulated annealing - attempt 1         
print('Example 1 - Attempt 1')
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
best_state, best_fitness, fitness_curve  = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10, 
                                                      max_iters = 1000, init_state = init_state,
                                                      random_state = 1)
print(best_state)
print(best_fitness)

# Solve using simulated annealing - attempt 2
best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 100, 
                                                      max_iters = 1000, init_state = init_state,
                                                      random_state = 1)
print('Example 1 - Attempt 2')
print(best_state)
print(best_fitness)

#  =================================================
# 
#  Example 2: 8-Queens Using Custom Fitness Function
# 
#  =================================================
# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    
    # Initialize counter
    fitness = 0
    
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):
                
                # If no attacks, then increment counter
                fitness += 1

    return fitness

# Check function is working correctly
state = np.array([1, 4, 1, 3, 5, 5, 2, 7])

# The fitness of this state should be 22
queens_max(state)

# Initialize custom fitness function object
fitness_cust = mlrose.CustomFitness(queens_max)

# Define optimization problem object
problem_cust = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_cust, maximize = True, max_val = 8)
# Solve using simulated annealing - attempt 1
best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem_cust, schedule = schedule, 
                                                      max_attempts = 10, max_iters = 1000, 
                                                      init_state = init_state, random_state = 1)
print('Example 2 - Attempt 1')
print(best_state)
print(best_fitness)
# Solve using simulated annealing - attempt 2
best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem_cust, schedule = schedule, 
                                                      max_attempts = 100, max_iters = 1000, 
                                                      init_state = init_state, random_state = 1)
print('Example 2 - Attempt 2')
print(best_state)
print(best_fitness)

#  =================================================
# 
# Example 3: Travelling Salesperson Using Coordinate-Defined Fitness Function
# 
#  =================================================
# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

# Initialize fitness function object using coords_list
fitness_coords = mlrose.TravellingSales(coords = coords_list)

# Define optimization problem object
problem_fit = mlrose.TSPOpt(length = 8, fitness_fn = fitness_coords, maximize = False)
# Solve using genetic algorithm - attempt 1
best_state, best_fitness, _ = mlrose.genetic_alg(problem_fit, random_state = 2)
print('Example 3 - Attempt 1')
print(best_state)
print(best_fitness)

# Solve using genetic algorithm - attempt 2
best_state, best_fitness, _ = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2, max_attempts = 100,
                                              random_state = 2)
print(best_state)
print(best_fitness)


#  =================================================
# 
# Example 4: Travelling Salesperson Using Distance-Defined Fitness Function
# 
#  =================================================