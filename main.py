from Part1 import Part1
# import subprocess as sp
# sp.call('clear',shell=True)

def runPart1():
    part1 = Part1()
    print('Running Random Hill-Climb...')
    part1.runRHC()
    # print('Running Simulated Annealing...')
    # part1.runSA()
    # print('Running Genetic Algorithm...')
    # part1.runGA()
    # print('Running MIMIC...')
    # part1.runMIMIC()

# Main block
runPart1()