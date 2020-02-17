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
    part1 = Part1(problem=fourPeaksProblem)

    part1.runAll()
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