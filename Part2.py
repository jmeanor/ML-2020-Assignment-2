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
        self.data = data2
        return data2

    # =====================================================================
    #   Splits the dataset into a training and test set.
    #   Source: https://mlrose.readthedocs.io/en/stable/source/tutorial3.html
    # =====================================================================
    def split_data(self):
         # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.data['data'], self.data['target'], \
                                                            test_size = 0.2, random_state = 3)
        X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, \
                                                            test_size = 0.2, random_state = 7)

        # Normalize feature data
        scaler = MinMaxScaler()
        self.X_train_scaled     = scaler.fit_transform(X_train)
        self.X_test_scaled      = scaler.transform(X_test)
        self.X_validate_scaled  = scaler.transform(X_validate)

        # One hot encode target values
        one_hot = OneHotEncoder()
        self.y_train_hot    = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
        self.y_test_hot     = one_hot.transform(y_test.reshape(-1, 1)).todense()
        self.y_validate_hot = one_hot.transform(y_validate.reshape(-1, 1)).todense()
    
    # =====================================================================
    # Based on tutorial Code from MLRose Docs
    # Source: https://mlrose.readthedocs.io/en/stable/source/tutorial3.html
    # 
    # =====================================================================
    def run(self):
        self.loadDataset()
        self.split_data()

        # Hyperparams - Trial One
        h_params = {
            'learning_rates': np.linspace(0.1, .5, 10),
            'max_iters': 100,
            'activation_functions': ['identity', 'relu', 'sigmoid', 'tanh'],
            'hidden_layers': [5],
            'restarts': 8
        }

        # Hyperparams - Trial Two
        h_params = {
            'learning_rates': np.linspace(0.3, .5, 10),
            'max_iters': 100,
            'activation_functions': ['identity', 'relu', 'tanh'],
            'hidden_layers': [[5], [10], [5,5]],
            
            # RHC
            'restarts': 8
        }
        self.runTrial('random_hill_climb', **h_params)

        # GA
        pop_size = [50, 100, 200]
        mutation_prob = np.linspace(0.1, 1, 5)
        
        return

        # =========================================================================================
        # Initialize neural network object and fit object
        # print('Training Gradient Descent')
        # nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [20,20], activation = 'relu', \
        #                                 algorithm = 'gradient_descent', max_iters = 1000, \
        #                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
        #                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
        #                                 random_state = 3)

        # nn_model2.fit(self.X_train_scaled, self.y_train_hot)

        # # Predict labels for train set and assess accuracy
        # y_train_pred = nn_model2.predict(self.X_train_scaled)

        # y_train_accuracy = accuracy_score(self.y_train_hot, y_train_pred)

        # print('Train: ', y_train_accuracy)
        # # 0.625

        # # Predict labels for test set and assess accuracy
        # y_test_pred = nn_model2.predict(self.X_test_scaled)

        # y_test_accuracy = accuracy_score(self.y_test_hot, y_test_pred)

        # print('Test: ', y_test_accuracy)
        # 0.566666666667

        # =================================================================================
        # Initialize neural network object and fit object
        # print('Training Genetic Algorithm')
        # nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [20,20], activation = 'relu', \
        #                                 algorithm = 'genetic_alg', max_iters = 1000, \
        #                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
        #                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
        #                                 random_state = 3)

        # nn_model2.fit(self.X_train_scaled, self.y_train_hot)

        # # Predict labels for train set and assess accuracy
        # y_train_pred = nn_model2.predict(self.X_train_scaled)

        # y_train_accuracy = accuracy_score(self.y_train_hot, y_train_pred)

        # print('Train: ', y_train_accuracy)
        # # 0.625

        # # Predict labels for test set and assess accuracy
        # y_test_pred = nn_model2.predict(self.X_test_scaled)

        # y_test_accuracy = accuracy_score(self.y_test_hot, y_test_pred)

        # print('Test: ', y_test_accuracy)
        # # 0.566666666667

    def runTrial(self, algorithm, **h_params):
        # Initialize neural network object and fit object
        print(h_params)

        activation_functions    = h_params['activation_functions']
        hidden_layers           = h_params['hidden_layers']
        learning_rates          = h_params['learning_rates']
        restarts                = h_params['restarts']
        max_iters               = h_params['max_iters']

        csvFile = open(os.path.join(self.savePath, algorithm+'_output.csv'), 'w')
        header = 'Algorithm, Activation Function, Learning Rate, Restarts, Hidden Layers, Training Accuracy, Testing Accuracy, Training Time\n'
        csvFile.write(header)

        for activation in activation_functions:
            for layers in hidden_layers:
                for learning_rate in learning_rates:
                    paramString = 'RHC, activation, %s, learning_rate, %f, restarts, 8, hidden_layers, %s' %(activation, learning_rate, layers)
                    log.info(paramString)
                    # print('Learning rate: ', learning_rate)
                    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = layers, 
                                                activation = activation, 
                                                algorithm = algorithm, 
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
                    nn_model1.fit(self.X_train_scaled, self.y_train_hot)
                    elapsed = time.process_time() - start
                    # log.info('\tElapsed time, %s' %elapsed)

                    # Predict labels for train set and assess accuracy
                    y_train_pred = nn_model1.predict(self.X_train_scaled)
                    y_train_accuracy = accuracy_score(self.y_train_hot, y_train_pred)

                    # Predict labels for test set and assess accuracy
                    y_test_pred = nn_model1.predict(self.X_test_scaled)
                    y_test_accuracy = accuracy_score(self.y_test_hot, y_test_pred)
                    # print('Test: ', y_test_accuracy)
                    log.info('\tTraining Accuracy,\t %f' %(y_train_accuracy))
                    log.info('\tTesting Accuracy,\t %f'%y_test_accuracy)
                    log.info('\tTraining Time,\t\t %f' %elapsed)
                    vals = '%s,%s,%s,%s,%s,%s,%s,%s,\n' %(algorithm, activation, learning_rate, restarts, layers, y_train_accuracy, y_test_accuracy, elapsed)
                    csvFile.write(vals)
                    # confusion = confusion_matrix(self.y_train_hot, y_train_pred)

        csvFile.close()

