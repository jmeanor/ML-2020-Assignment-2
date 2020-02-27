import os
from matplotlib import pyplot as plt
import mlrose_reborn as mlrose
import numpy as np
import pydash as _ 

import graph

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Initialize fitness function object using pre-defined class
# fitness = mlrose.Queens()
# fitness = mlrose.FourPeaks(t_pct=0.15)

# Solve using simulated annealing - attempt 1
# init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# init_state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])

# Define optimization problem object
# problem = mlrose.DiscreteOpt(length=len(
#     init_state), fitness_fn=fitness, maximize=False, max_val=len(init_state))

# Define decay schedule
schedule = mlrose.GeomDecay()


class Part1():
    # Tutorial from MLRose Documentation
    # https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb

    def __init__(self, name, fitness, init_state, problem, schedule=schedule):
        self.name = name
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

    # Convenience function to run everything.
    def runAll(self, savePath):
        a = np.array(self.runRHC())
        b = np.array(self.runSA())
        c = np.array(self.runGA())
        d = np.array(self.runMIMIC())
        maxLen = max((len(a), len(b), len(c), len(d)))
        arr = np.zeros((4, maxLen))
        
        arr[0, :len(a)] = a
        arr[1, :len(b)] = b
        arr[2, :len(c)] = c
        arr[3, :len(d)] = d

        # print('----------------')
        # print('RunAll Result: ')
        # print(arr)
        saveDir = os.path.join(savePath, '%s.png' %self.name)
        graph.plotPart1(arr, saveDir, title=self.name, xmax=np.max(arr[0]) + 5)

    # Random Hill-Climbing
    def runRHC(self):
        default = {
            'problem': self.problem,
            'max_attempts': 10, 
            'max_iters': np.inf, 
            'init_state': self.init_state,
            'curve': True, 
            'random_state': 1
        }

        maxAttempts = [5, 25, 50, 75, 100, 125, 150, 200, 225, 250, 275, 300, 350, 400, 450, 500]
        random = [1, 19, 49, 512903123]
        bestFitness = None
        (bestState, bestCurve, bestParams) = None, None, None
        for i in maxAttempts:
            for j in random:
                params = _.assign({}, default, {'max_attempts': i, 'random_state': j})
                state, fitness, curve = self._run(mlrose.random_hill_climb, name='%s' %i, **params)
                if bestFitness == None or fitness < bestFitness:
                    bestFitness = fitness
                    (bestState, bestCurve, bestParams) = state, curve, params
                if fitness == 0:
                    break
        
        print('RHC - Best fitness found on max_attempt %s, random_state: %s' %(bestParams['max_attempts'], bestParams['random_state']))
        # print('Curve: %s' %curve)
        # print('RHC - Params: %s' %params)
        return bestCurve

    # Simulated Annealing
    def runSA(self):
        default = {
            'problem': self.problem, 
            'schedule': self.schedule, 
            'max_attempts': 10,
            'max_iters': 100, 
            'init_state': self.init_state,
            'curve': True, 
            'random_state': 1
        }
        # state, fitness, curve = self._run(mlrose.simulated_annealing, name='1', **default)

        maxAttempts = [5, 25, 50, 75, 100, 125, 150, 200, 225, 250, 275, 300, 350, 400, 450, 500]
        schedules = [mlrose.GeomDecay(), mlrose.ExpDecay(), mlrose.ArithDecay()]
        bestFitness = None
        (bestState, bestCurve, bestParams) = None, None, None
        for i in maxAttempts:
            for j in schedules:
                params = _.assign({}, default, {'max_attempts': i, 'schedule': j}) 
                state, fitness, curve = self._run(mlrose.simulated_annealing, name='%s' %i, **params)
                if bestFitness == None or fitness < bestFitness:
                    bestFitness = fitness
                    (bestState, bestCurve, bestParams) = state, curve, params
                if fitness == 0:
                    break
        # print('SA - Best fitness found on max_attempt %s' %bestParams['max_attempts'])
        print('SA - Params: %s' %bestParams)

        return bestCurve

    # Genetic Algorithms
    def runGA(self):
        default =  {
            'problem': self.problem, 
            'pop_size': 200,
            'mutation_prob': 0.1,
            'max_attempts': 10,
            'max_iters': np.inf,
            'curve': True,
            'random_state': None
        }

        state, fitness, curve = self._run(mlrose.genetic_alg, name='1', **default)
        return curve

    # Mimic
    def runMIMIC(self):
        default = {
            'problem': self.problem, 
            'pop_size': 200, 
            'keep_pct': 0.2, 
            'max_attempts':10, 
            'max_iters': np.inf, 
            'curve': True, 
            'random_state': None
        }
        # Experimental
        self.problem.set_mimic_fast_mode(True)

        state, fitness, curve = self._run(mlrose.mimic, name='1', **default)
        return curve
    
    def _run(self, algorithm, name='', **kwargs):
        # print('%s attempt %s' %(algorithm.__name__, name))
        best_state, best_fitness, fitness_curve = algorithm(**kwargs)

        # print(best_state)
        # print(best_fitness)
        # print(fitness_curve)

        return best_state, best_fitness, fitness_curve