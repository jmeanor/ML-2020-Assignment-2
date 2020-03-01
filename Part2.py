from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import mlrose_hiive as mlrose
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import time, os

# Logging
import logging
log = logging.getLogger()

class Part2():

    # Constructor 
    def __init__(self, savePath):
        self.savePath = savePath
    
    def loadDataset(self):
        df = pd.read_csv('./input/hirosaki_temp_cherry.csv', delimiter=',', header=0)
        
        # pprint(df)
        target = np.array(df['flower_status'])
        data = np.array(df.drop('flower_status', axis=1))
        data2 = {
            'data': data,
            'target': target
        }
        return data2

    def run(self):
        # random_hill_climb’, ‘simulated_annealing’, ‘genetic_alg’ or ‘gradient_descent’

        # Tutorial Code from MLRose Docs
        # Source: https://mlrose.readthedocs.io/en/stable/source/tutorial3.html
        # 
        # Load the Iris dataset
        # data = load_iris()
        data = self.loadDataset()

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], \
                                                            test_size = 0.2, random_state = 3)

        # Normalize feature data
        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # One hot encode target values
        one_hot = OneHotEncoder()

        y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
        y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

        # print(len(y_train_hot))
        # print(len(y_test_hot))

        # Hyperparams
        learning_rates = np.linspace(0.1, .5, 10) # 0.4111111111111111
        # learning_rates = [0.4]
        max_iters = 100
        activation_functions = ['identity', 'relu', 'sigmoid', 'tanh']
        hidden_layers = [5]
        
        # RHC
        restarts = 8
        # GA
        pop_size = [50, 100, 200]
        mutation_prob = np.linspace(0.1, 1, 5)

        # Initialize neural network object and fit object
        print('Training Random Hill Climb')
        csvFile = open(os.path.join(self.savePath, 'RHC_output.csv'), 'w')
        header = 'Algorithm, Activation Function, Learning Rate, Restarts, Hidden Layers, Training Accuracy, Testing Accuracy, Training Time\n'
        csvFile.write(header)

        for activation in activation_functions:
            for learning_rate in learning_rates:
                paramString = 'RHC, activation, %s, learning_rate, %f, restarts, 8, hidden_layers, %s' %(activation, learning_rate, hidden_layers)
                log.info(paramString)
                # print('Learning rate: ', learning_rate)
                nn_model1 = mlrose.NeuralNetwork(hidden_nodes = hidden_layers, 
                                            activation = activation, 
                                            algorithm = 'random_hill_climb', 
                                            restarts=restarts,
                                            max_iters = max_iters,
                                            bias = True, 
                                            is_classifier = True, 
                                            learning_rate = learning_rate, 
                                            early_stopping = True, 
                                            clip_max = 5,
                                            curve=True, 
                                            max_attempts = 100,
                                            random_state = 3)
                start = time.process_time()
                nn_model1.fit(X_train_scaled, y_train_hot)
                elapsed = time.process_time() - start
                # log.info('\tElapsed time, %s' %elapsed)

                # Predict labels for train set and assess accuracy
                y_train_pred = nn_model1.predict(X_train_scaled)
                y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

                # Predict labels for test set and assess accuracy
                y_test_pred = nn_model1.predict(X_test_scaled)
                y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
                # print('Test: ', y_test_accuracy)
                log.info('\tTraining Accuracy,\t %f' %(y_train_accuracy))
                log.info('\tTesting Accuracy,\t %f'%y_test_accuracy)
                log.info('\tTraining Time,\t\t %f' %elapsed)
                # 'RHC, activation, %s, learning_rate, %f, restarts, 8, hidden_layers, %s' %(activation, learning_rate, hidden_layers)
                vals = '%s,%s,%s,%s,%s,%s,%s,%s,\n' %('RHC', activation, learning_rate, restarts, hidden_layers, y_train_accuracy, y_test_accuracy, elapsed)
                csvFile.write(vals)
                # confusion = confusion_matrix(y_train_hot, y_train_pred)

        csvFile.close()
        return

        # =========================================================================================
        # Initialize neural network object and fit object
        print('Training Gradient Descent')
        nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [20,20], activation = 'relu', \
                                        algorithm = 'gradient_descent', max_iters = 1000, \
                                        bias = True, is_classifier = True, learning_rate = 0.0001, \
                                        early_stopping = True, clip_max = 5, max_attempts = 100, \
                                        random_state = 3)

        nn_model2.fit(X_train_scaled, y_train_hot)

        # Predict labels for train set and assess accuracy
        y_train_pred = nn_model2.predict(X_train_scaled)

        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

        print('Train: ', y_train_accuracy)
        # 0.625

        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model2.predict(X_test_scaled)

        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        print('Test: ', y_test_accuracy)
        # 0.566666666667

        # =================================================================================
        # Initialize neural network object and fit object
        print('Training Genetic Algorithm')
        nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [20,20], activation = 'relu', \
                                        algorithm = 'genetic_alg', max_iters = 1000, \
                                        bias = True, is_classifier = True, learning_rate = 0.0001, \
                                        early_stopping = True, clip_max = 5, max_attempts = 100, \
                                        random_state = 3)

        nn_model2.fit(X_train_scaled, y_train_hot)

        # Predict labels for train set and assess accuracy
        y_train_pred = nn_model2.predict(X_train_scaled)

        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

        print('Train: ', y_train_accuracy)
        # 0.625

        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model2.predict(X_test_scaled)

        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        print('Test: ', y_test_accuracy)
        # 0.566666666667


