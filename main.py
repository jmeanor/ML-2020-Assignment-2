import mlrose_hiive as mlrose
import numpy as np
import os
import errno
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


def setLog(path, oldHandler=None):
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
    init_state = None
    fourPeaksProblem = mlrose.DiscreteOpt(length=12,
        fitness_fn=fitness, maximize=True, max_val=2)

    part1_1 = Part1(name='Four Peaks', fitness=fitness,
                    problem=fourPeaksProblem, init_state=init_state)
    part1_1.runAll(savePath)

    fitness = mlrose.Queens()
    init_state = None
    eightQueensProblem = mlrose.DiscreteOpt(length=8,
        fitness_fn=fitness, maximize=False, max_val=8)
    part1_2 = Part1(name='Eight Queens', fitness=fitness,
                    problem=eightQueensProblem, init_state=init_state)
    part1_2.runAll(savePath)

    fitness = mlrose.SixPeaks(t_pct=0.15)
    init_state = None
    sixPeaksProblem = mlrose.DiscreteOpt(length=11,
        fitness_fn=fitness, maximize=True, max_val=2)
    part1_4 = Part1(name='Six Peaks', fitness=fitness,
                    problem=sixPeaksProblem, init_state=init_state)
    part1_4.runAll(savePath)

    fitness = mlrose.FlipFlop()
    init_state = None
    flipFlopProblem = mlrose.DiscreteOpt(length=7,
        fitness_fn=fitness, maximize=True, max_val=2)
    part1_5 = Part1(name='Flip Flop - 7', fitness=fitness,
                    problem=flipFlopProblem, init_state=init_state)
    part1_5.runAll(savePath)


    fitness = mlrose.FlipFlop()
    init_state = None
    flipFlopProblem = mlrose.DiscreteOpt(length=100,
        fitness_fn=fitness, maximize=True, max_val=2)
    part1_5 = Part1(name='Flip Flop - 100', fitness=fitness,
                    problem=flipFlopProblem, init_state=init_state)
    part1_5.runAll(savePath)

    fitness = mlrose.Queens()
    init_state = None
    eightQueensProblem = mlrose.DiscreteOpt(length=80,
        fitness_fn=fitness, maximize=False, max_val=8)
    part1_2 = Part1(name='Eighty Queens', fitness=fitness,
                    problem=eightQueensProblem, init_state=init_state)
    part1_2.runAll(savePath)

    fitness = mlrose.FlipFlop()
    init_state = None
    flipFlopProblem = mlrose.DiscreteOpt(length=15,
        fitness_fn=fitness, maximize=True, max_val=2)
    part1_5 = Part1(name='Flip Flop - 15', fitness=fitness,
                    problem=flipFlopProblem, init_state=init_state)
    part1_5.runAll(savePath)


    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
    fitness = mlrose.MaxKColor(edges)
    init_state = None
    maxKColorsProblem = mlrose.DiscreteOpt(length=7,
        fitness_fn=fitness, maximize=False, max_val=2)
    part1_3 = Part1(name='Max-K Color', fitness=fitness,
                    problem=maxKColorsProblem, init_state=init_state)
    part1_3.runAll(savePath)

    # =============================================================
    #  Source - Tutorial from MLRose Docs
    #  https://mlrose.readthedocs.io/en/stable/source/tutorial2.html
    # 
    # =============================================================
    # Create list of city coordinates
    coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

    # Initialize fitness function object using coords_list
    fitness_coords = mlrose.TravellingSales(coords = coords_list)

    # Create list of distances between pairs of cities
    dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), \
                (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), \
                (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), \
                (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), \
                (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
                (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), \
                (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]

    # Initialize fitness function object using dist_list
    fitness_dists = mlrose.TravellingSales(distances = dist_list)

    # Define optimization problem object
    problem_fit = mlrose.TSPOpt(length = 8, fitness_fn = fitness_coords, maximize=False)

    coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

    # Define optimization problem object
    problem_no_fit = mlrose.TSPOpt(length = 8, coords = coords_list, maximize=False)

    part1_6 = Part1(name='TSP', fitness=coords_list,
                    problem=problem_no_fit, init_state=None)
    part1_6.runAll(savePath)

    # Knapsack
    weights = np.random.randint(2, high=20, size=50)
    values = np.random.randint(2, high=100, size=50)
    max_weight_pct = 0.8
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    knapsackProblem = mlrose.DiscreteOpt(length=50,
        fitness_fn=fitness, maximize=False, max_val=2)

    part1_7 = Part1(name='Knapsack', fitness=fitness,
                    problem=knapsackProblem, init_state=None)
    part1_7.runAll(savePath)


    # Debug
    # print('Running Random Hill-Climb...\n')
    # part1.runRHC()
    # print('Running Simulated Annealing...\n')
    # part1.runSA()
    # print('Running Genetic Algorithm...\n')
    # part1.runGA()
    # print('Running MIMIC...\n')
    # part1.runMIMIC()


def runPart2(savePath):
    from Part2 import Part2

    part2 = Part2(savePath)
    part2.run()


# Main block
# timestamp = datetime.now().strftime('%b-%d-%y %I:%M:%S %p')
# path1 = createDateFolder((timestamp, "Part-1"))
# part1Handler = setLog(path1)

# runPart1(path1)
# runPart1_2(path1)

runPart2('')
