import mlrose_reborn as mlrose
import numpy as np
import os, errno
from datetime import datetime

# Assignment Code
from Part1 import Part1

# Logging
import myLogger
import logging
logger = logging.getLogger()
logger.info('Initializing main.py')
log = logging.getLogger()

###
#    source: https://stackoverflow.com/questions/14115254/creating-a-folder-with-timestamp/14115286
###
def createDateFolder(suffix=("")):
    mydir = os.path.join(os.getcwd(), 'output', *suffix)
    # print('mydir %s' %mydir)
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir

###
#    source: My assignment 1 code
###
def setLog(path, oldHandler = None):
    if oldHandler != None:
        myLogger.logger.removeHandler(oldHandler)
    logPath = os.path.join(path, 'metadata.txt')
    fh = myLogger.logging.FileHandler(logPath)
    fh.setLevel(logging.INFO)
    fmtr = logging.Formatter('%(message)s')
    fh.setFormatter(fmtr)
    myLogger.logger.addHandler(fh)
    return fh

def runPart1(savePath):
    fitness = mlrose.FourPeaks(t_pct=0.15)
    init_state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    fourPeaksProblem = mlrose.DiscreteOpt(length=len(
                                    init_state), fitness_fn=fitness, maximize=False, max_val=len(init_state))

    part1_1 = Part1(name='Four Peaks', fitness=fitness, problem=fourPeaksProblem, init_state=init_state)
    part1_1.runAll(savePath)

    fitness = mlrose.Queens()
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    eightQueensProblem = mlrose.DiscreteOpt(length=len(
                                    init_state), fitness_fn=fitness, maximize=False, max_val=len(init_state))
    part1_2 = Part1(name='Eight Queens', fitness=fitness, problem=eightQueensProblem, init_state=init_state)
    part1_2.runAll(savePath)

    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
    fitness = mlrose.MaxKColor(edges)
    init_state = np.array([0, 1, 0, 1, 1])
    eightQueensProblem = mlrose.DiscreteOpt(length=len(
                                    init_state), fitness_fn=fitness, maximize=False, max_val=len(init_state))
    part1_3 = Part1(name='Max-K Color', fitness=fitness, problem=eightQueensProblem, init_state=init_state)
    part1_3.runAll(savePath)


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
timestamp = datetime.now().strftime('%b-%d-%y %I:%M:%S %p')
path1 = createDateFolder((timestamp, "Part-1"))
part1Handler = setLog(path1)

runPart1(path1)