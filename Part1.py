import mlrose_reborn as mlrose
import numpy as np
import pydash as _ 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Initialize fitness function object using pre-defined class
# fitness = mlrose.Queens()
fitness = mlrose.FourPeaks(t_pct=0.15)

# Solve using simulated annealing - attempt 1
# init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
init_state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])

# Define optimization problem object
problem = mlrose.DiscreteOpt(length=len(
    init_state), fitness_fn=fitness, maximize=False, max_val=len(init_state))

# Define decay schedule
schedule = mlrose.ExpDecay()


class Part1():
    # Tutorial from MLRose Documentation
    # https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb

    def __init__(self, fitness=fitness, init_state=init_state, problem=problem, schedule=schedule):
        # Initialize fitness function object using pre-defined class
        # fitness = mlrose.Queens()
        self.fitness = fitness

        # Solve using simulated annealing - attempt 1
        # init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        self.init_state = init_state

        # Define optimization problem object
        self.problem = problem

        # Define decay schedule
        self.schedule = schedule

    # Random Hill-Climbing
    def runRHC(self):
        default = {
            'problem': self.problem,
            'max_attempts': 10, 
            'max_iters': np.inf, 
            'init_state': self.init_state,
            'random_state': 1
        }
        self._rhc('1', **default)
        
        a = _.assign({}, default, {'max_iters': 10})
        self._rhc('a', **a)
        
        b = _.assign({}, default, {'max_iters': 20})
        self._rhc('b', **b)


    def _rhc(self, name = '', **kwargs):
        print('RHC Trial %s'  %name )

        best_state, best_fitness, _ = mlrose.random_hill_climb( **kwargs )

        print(best_state)
        print(best_fitness)

    # Simulated Annealing
    def runSA(self):
        default = {
            'problem': self.problem, 
            'schedule': self.schedule, 
            'max_attempts': 10,
            'max_iters': 1000, 
            'init_state': self.init_state,
            'random_state': 1
        }

        self._sa('1', **default)
        
    def _sa(self, name='', **kwargs):
        print('SA Attempt %s' %name)
        best_state, best_fitness, _ = mlrose.simulated_annealing(**kwargs)
        print(best_state)
        print(best_fitness)

    # Genetic Algorithms
    def runGA(self):
        default = {
            'problem': self.problem, 
            'max_attempts': 10, 
            'max_iters': np.inf, 
            'init_state': self.init_state,
            'random_state': 1
        }

        self._ga('1', **default)
        

    def _ga(self, name='', **kwargs):
        best_state, best_fitness, _ = mlrose.random_hill_climb(**kwargs)
        print('GA Attempt 1 ')
        print(best_state)
        print(best_fitness)

    # Mimic
    def runMIMIC(self):
        best_state, best_fitness, _ = mlrose.genetic_alg(
            self.problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf, random_state=1)
        print('MIMIC Attempt 1 ')
        print(best_state)
        print(best_fitness)
    
