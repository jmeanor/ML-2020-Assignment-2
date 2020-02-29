from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("macOSX")

import numpy as np

def plotPart1(arr, saveDir, title='', xlabel='', ylabel='', isMaximizing=True, xmax=50):
    fig = plt.figure()
    fig.add_axes()
    ax1 = fig.add_subplot(111)
    
    arr = arr[:, 0:xmax]
    # ax1.plot(arr[0], color='b', marker='.', linewidth=1, label='Random Hill-Climb')
    # ax1.plot(arr[1], color='g', marker='*', linewidth=1, label='Simulated Annealing')
    # ax1.plot(arr[2], color='m', marker='x', linewidth=1, label='Genetic Algorithm')
    # ax1.plot(arr[3], color='y', marker='h', linewidth=1, label='MIMIC')
    ax1.plot(arr[0], color='b', marker=None, linewidth=1, label='Random Hill-Climb')
    ax1.plot(arr[1], color='g', marker=None, linewidth=1, label='Simulated Annealing')
    ax1.plot(arr[2], color='m', marker=None, linewidth=1, label='Genetic Algorithm')
    ax1.plot(arr[3], color='y', marker=None, linewidth=1, label='MIMIC')

    ylabel = ('Fitness Score' if isMaximizing else 'Fitness Score Error')

    ax1.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax1.legend(loc='best')
    plt.grid(True)
    plt.savefig(saveDir, bbox_inches='tight')

    # plt.show()

def plotPart1_2(arr, saveDir, title='', xlabel='', ylabel='', isMaximizing=True, xmax=50):
    fig = plt.figure()
    fig.add_axes()
    ax1 = fig.add_subplot(111)
    
    arr = arr[:, 0:xmax]
    # ax1.plot(arr[0], color='b', marker='.', linewidth=1, label='Random Hill-Climb')
    # ax1.plot(arr[1], color='g', marker='*', linewidth=1, label='Simulated Annealing')
    # ax1.plot(arr[2], color='m', marker='x', linewidth=1, label='Genetic Algorithm')
    # ax1.plot(arr[3], color='y', marker='h', linewidth=1, label='MIMIC')
    ax1.plot(arr[0], color='b', marker=None, linewidth=1, label='Random Hill-Climb')
    ax1.plot(arr[1], color='g', marker=None, linewidth=1, label='Simulated Annealing')
    ax1.plot(arr[2], color='m', marker=None, linewidth=1, label='Genetic Algorithm')
    ax1.plot(arr[3], color='y', marker=None, linewidth=1, label='MIMIC')

    ylabel = ('Fitness Score' if isMaximizing else 'Fitness Score Error')

    ax1.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax1.legend(loc='best')
    plt.grid(True)
    plt.savefig(saveDir, bbox_inches='tight')
