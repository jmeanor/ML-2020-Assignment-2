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
schedule = mlrose.GeomDecay()


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
        self._run(mlrose.random_hill_climb, name='1', **default)
        
        A = _.assign({}, default, {'max_iters': 10})
        self._run(mlrose.random_hill_climb, name='A', **A)
        
        B = _.assign({}, default, {'max_iters': 20})
        self._run(mlrose.random_hill_climb, name='B', **B)

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
        self._run(mlrose.simulated_annealing, name='1', **default)

    # Genetic Algorithms
    def runGA(self):
        default =  {
            'problem': self.problem, 
            'pop_size': 200, 
            'mutation_prob': 0.1, 
            'max_attempts': 10, 
            'max_iters': np.inf, 
            'curve': False, 
            'random_state': None
        }

        self._run(mlrose.genetic_alg, name='1', **default)


    # Mimic
    def runMIMIC(self):
        default = {
            'problem': self.problem, 
            'pop_size': 200, 
            'keep_pct': 0.2, 
            'max_attempts':10, 
            'max_iters': np.inf, 
            'curve': False, 
            'random_state': None, 
            'fast_mimic': False
        }

        self._run(mlrose.mimic, name='1', **default)
    
    def _run(self, algorithm, name='', **kwargs):
        print('%s attempt %s' %(algorithm.__name__, name))
        best_state, best_fitness, _ = algorithm(**kwargs)

        print(best_state)
        print(best_fitness)