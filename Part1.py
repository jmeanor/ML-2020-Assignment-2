import os
from matplotlib import pyplot as plt
import mlrose_hiive as mlrose
import numpy as np
import pydash as _

import graph
from pprint import pprint

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Logging
import logging
log = logging.getLogger()

# Define decay schedule
schedule = mlrose.GeomDecay()


class Part1():
    # Tutorial from MLRose Documentation
    # https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb

    def __init__(self, name, fitness, init_state, problem, schedule=schedule):
        self.name = name
        self.fitness = fitness
        self.init_state = init_state

        # Define optimization problem object
        self.problem = problem
        # Infer Maxmimize or Minimize
        self.isMaximize = (True if self.problem.get_maximize() > 0 else False)

        # Define decay schedule
        self.schedule = schedule

    # Convenience function to run everything.
    def runAll(self, savePath):
        log.info('%s:' % self.name)

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

        arr[0, len(a):maxLen] = a[-1]
        arr[1, len(b):maxLen] = b[-1]
        arr[2, len(c):maxLen] = c[-1]
        arr[3, len(d):maxLen] = d[-1]

        saveDir = os.path.join(savePath, '%s.png' % self.name)
        graph.plotPart1(arr, saveDir, title=self.name, isMaximizing=self.isMaximize, xmax=maxLen+5)

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

        maxAttempts = [5, 10, 20]
        restarts = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        bestFitness = None
        (bestState, bestCurve, bestParams) = None, None, None
        for i in maxAttempts:
            for j in restarts:
                params = _.assign(
                    {}, default, {'max_attempts': i, 'restarts': j})

                scores = []
                for r in range(5):
                    randomSeed = np.random.randint(0, 1000)
                    params = _.assign(
                        {}, params, {'random_state': randomSeed})
                    state, fitness, curve = self._run(
                        mlrose.random_hill_climb, name='%s' % i, **params)
                    scores.append(fitness)
                avgFitness = np.mean(scores)

                if bestFitness == None or (self.isMaximize and avgFitness > bestFitness) or (not self.isMaximize and avgFitness < bestFitness):
                    bestFitness = avgFitness
                    (bestState, bestCurve, bestParams) = state, curve, params
                # if fitness == 0:
                #     break

        print('RHC - Best fitness found on max_attempts: %s restarts: %s' %
              (bestParams['max_attempts'], bestParams['restarts']))
        log.info('\tRHC - Best fitness found: %s\n\t\tmax_attempts: %s \n\t\trestarts: %s' %
                 (bestFitness, bestParams['max_attempts'], bestParams['restarts']))

        return bestCurve

    # Simulated Annealing
    def runSA(self):
        default = {
            'problem': self.problem,
            'schedule': self.schedule,
            'max_attempts': 10,
            'max_iters': 1000,
            'init_state': self.init_state,
            'curve': True,
            'random_state': 1
        }

        maxAttempts = [5, 10, 20]
        schedules = [mlrose.GeomDecay(), mlrose.ExpDecay(),
                     mlrose.ArithDecay()]
        bestFitness = None
        (bestState, bestCurve, bestParams) = None, None, None
        for i in maxAttempts:
            for j in schedules:
                params = _.assign(
                    {}, default, {'max_attempts': i, 'schedule': j})

                scores = []
                for r in range(5):
                    randomSeed = np.random.randint(0, 1000)
                    params = _.assign(
                        {}, params, {'random_state': randomSeed})
                    state, fitness, curve = self._run(
                        mlrose.simulated_annealing, name='%s' % i, **params)
                    scores.append(fitness)
                avgFitness = np.mean(scores)

                if bestFitness == None or (self.isMaximize and avgFitness > bestFitness) or (not self.isMaximize and avgFitness < bestFitness):
                    bestFitness = avgFitness
                    (bestState, bestCurve, bestParams) = state, curve, params
                # if fitness == 0:
                #     break
        print('SA - Params: %s' % bestParams)
        log.info('\tSA - Best fitness found: %s\n\t\tmaxAttempts: %s \n\t\tschedule: %s' %
                 (bestFitness, bestParams['max_attempts'], type(bestParams['schedule']).__name__))

        return bestCurve

    # Genetic Algorithms
    def runGA(self):
        default = {
            'problem': self.problem,
            'pop_size': 200,
            'mutation_prob': 0.1,
            'max_attempts': 10,
            'max_iters': 100,
            'curve': True,
            'random_state': None
        }

        mutation_prob = np.linspace(0.1, 1, 5)
        pop_size = [50,100, 200]
        bestFitness = None
        for i in mutation_prob:
            for j in pop_size:
                params = _.assign(
                    {}, default, {'mutation_prob': i, 'pop_size': j})

                scores = []
                for r in range(5):
                    print('Running GA %i' %r)
                    randomSeed = np.random.randint(0, 1000)
                    params = _.assign(
                        {}, params, {'random_state': randomSeed})
                    state, fitness, curve = self._run(
                        mlrose.genetic_alg, name='%s' % i, **params)
                    scores.append(fitness)
                avgFitness = np.mean(scores)

                if bestFitness == None or (self.isMaximize and avgFitness > bestFitness) or (not self.isMaximize and avgFitness < bestFitness):
                    bestFitness = avgFitness
                    (bestState, bestCurve, bestParams) = state, curve, params
                # if fitness == 0:
                #     break
        log.info('\tGA - Best fitness found: %s\n\t\tmutation_prob: %s \n\t\tpop_size: %s' %
                 (bestFitness, bestParams['mutation_prob'], bestParams['pop_size']))

        return bestCurve

    # Mimic
    def runMIMIC(self):
        default = {
            'problem': self.problem,
            'pop_size': 200,
            'keep_pct': 0.2,
            'max_attempts': 10,
            'max_iters': np.inf,
            'curve': True,
            'random_state': None
        }
        # Experimental
        self.problem.set_mimic_fast_mode(True)

        state, fitness, curve = self._run(mlrose.mimic, name='1', **default)

        keep_pct = np.linspace(0.1, 1, 5)
        pop_size = [50, 100, 200]
        bestFitness = None
        for i in keep_pct:
            for j in pop_size:
                params = _.assign(
                    {}, default, {'keep_pct': i, 'pop_size': j})

                scores = []
                for r in range(5):
                    print('Running MIMIC %i' %r)
                    randomSeed = np.random.randint(0, 1000)
                    params = _.assign(
                        {}, params, {'random_state': randomSeed})
                    state, fitness, curve = self._run(
                        mlrose.mimic, name='%s' % i, **params)
                    scores.append(fitness)
                avgFitness = np.mean(scores)

                if bestFitness == None or (self.isMaximize and avgFitness > bestFitness) or (not self.isMaximize and avgFitness < bestFitness):
                    bestFitness = avgFitness
                    (bestState, bestCurve, bestParams) = state, curve, params
                # if fitness == 0:
                #     break
        log.info('\tMIMIC - Best fitness found: %s\n\t\tkeep_pct: %s \n\t\tpop_size: %s' %
                 (bestFitness, bestParams['keep_pct'], bestParams['pop_size']))
        # log.info('MIMIC CURVE: %s' %bestCurve)

        return bestCurve

    def _run(self, algorithm, name='', **kwargs):
        # print('%s attempt %s' %(algorithm.__name__, name))
        best_state, best_fitness, fitness_curve = algorithm(**kwargs)

        # print(best_state)
        # print(best_fitness)
        # print(fitness_curve)

        return best_state, best_fitness, fitness_curve
