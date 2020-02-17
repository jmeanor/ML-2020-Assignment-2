from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("macOSX")

import numpy as np

def plotPart1(arr, title='', xlabel='', ylabel='' ):
    fig = plt.figure()
    fig.add_axes()
    ax1 = fig.add_subplot(111)
    
    arr = arr[:, 0:40]
    ax1.plot(arr[0], color='b', marker='.', linewidth=1, label='Random Hill-Climb')
    ax1.plot(arr[1], color='g', marker='*', linewidth=1, label='Simulated Annealing')
    ax1.plot(arr[2], color='m', marker='x', linewidth=1, label='Genetic Algorithm')
    ax1.plot(arr[3], color='y', marker='h', linewidth=1, label='MIMIC')
    
    ax1.set(title=title, ylabel='Fitness Error', xlabel='Iterations')
    ax1.legend(loc='best')
    plt.grid(True)

    plt.show()