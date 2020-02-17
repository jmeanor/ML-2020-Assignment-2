import mlrose_reborn as mlrose
import numpy as np

# Assignment Code
from Part1 import Part1

# Logging
import myLogger
import logging
logger = logging.getLogger()
logger.info('Initializing main.py')
log = logging.getLogger()

def runPart1():
    fitness = mlrose.FourPeaks(t_pct=0.15)
    init_state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    fourPeaksProblem = mlrose.DiscreteOpt(length=len(
                                    init_state), fitness_fn=fitness, maximize=False, max_val=len(init_state))

    part1_1 = Part1(name='Four Peaks', fitness=fitness, problem=fourPeaksProblem, init_state=init_state)
    part1_1.runAll()

    fitness = mlrose.Queens()
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    eightQueensProblem = mlrose.DiscreteOpt(length=len(
                                    init_state), fitness_fn=fitness, maximize=False, max_val=len(init_state))
    part1_2 = Part1(name='Eight Queens', fitness=fitness, problem=eightQueensProblem, init_state=init_state)
    part1_2.runAll()


    # Debug

    # print('Running Random Hill-Climb...\n')
    # part1.runRHC()
    # print('Running Simulated Annealing...\n')
    # part1.runSA()
    # print('Running Genetic Algorithm...\n')
    # part1.runGA()
    # print('Running MIMIC...\n')
    # part1.runMIMIC()

# Main block
runPart1()