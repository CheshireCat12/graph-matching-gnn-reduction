import os
from collections import defaultdict
from itertools import product
from pathlib import Path
from time import time

import numpy as np
cimport numpy as np
import pandas as pd
from progress.bar import Bar

from graph_pkg.experiment.runner import Runner
from graph_pkg.algorithm.knn cimport KNNClassifier
from graph_pkg.utils.functions.helper import calc_accuracy
from graph_gnn_embedding.utils.coordinator.coordinator_gnn_embedding_classifier import CoordinatorGNNEmbeddingClassifier

class RunnerKNNGNN(Runner):

    def __init__(self, parameters, logger):
        super(RunnerKNNGNN, self).__init__(parameters)
        self.logger = logger

    def run(self):
        print('Run KNN with Reduced Graphs')

        # Init the graph gatherer
        coordinator_params = self.parameters.current_coordinator

        self.save_stats('The code is running\n', 'log.txt', save_params=False)

        self.coordinator = CoordinatorGNNEmbeddingClassifier(**coordinator_params)

        self.save_stats('The graphs are loaded\n', 'log.txt', save_params=False)

        if self.parameters.optimize:
            best_params = self.optimization()
            # best_params = (1, 0.7)

            self.evaluate(best_params)


    def optimization(self):
        cdef:
            KNNClassifier knn

        ##################
        # Set parameters #
        ##################

        num_cores = self.parameters.num_cores
        parallel = num_cores > 0

        graphs_train, labels_train = self.coordinator.train_split()
        graphs_val, labels_val = self.coordinator.val_split()

        knn = KNNClassifier(self.coordinator.ged, parallel, verbose=False)
        knn.train(graphs_train=graphs_train, labels_train=labels_train)

        # Hyperparameters to tune
        alpha_start, alpha_end, alpha_step = [type_(val)
                                              for val, type_ in zip(self.parameters.alpha, [int, int, float])]
        alphas = [alpha_step * i for i in range(alpha_start, alpha_end)]
        ks = self.parameters.ks

        self.logger.data['best_acc'] = float('-inf')
        self.logger.data['best_params'] = (None, None)
        self.logger.data['hyperparameters_tuning'] = {
            'alphas': alphas,
            'ks':ks
        }
        self.logger.data['accuracies'] = defaultdict(list)
        self.logger.data['prediction_times'] = []

        hyperparameters = product(ks, alphas)
        len_hyperparameters = len(ks) * len(alphas)

        bar = Bar(f'Processing Graphs : Optimization', max=len_hyperparameters)

        for k_param, alpha in hyperparameters:
            alpha = round(alpha, 2)
            self.coordinator.edit_cost.update_alpha(alpha)

            start_time = time()
            predictions = knn.predict(graphs_pred=graphs_val, k=k_param, num_cores=num_cores)
            prediction_time = time() - start_time

            self.logger.data['prediction_times'].append(prediction_time)

            acc = calc_accuracy(np.array(labels_val, dtype=np.int32), predictions)

            if acc >= self.logger.data['best_acc']:
                self.logger.data['best_acc'] = acc
                self.logger.data['best_params'] = (k_param, alpha)

            self.logger.data['accuracies'][k_param].append(acc)

            self.logger.save_data()

            Bar.suffix = f'%(index)d/%(max)d | ' \
                         f'Current acc: {acc:.2f} ({k_param}, {alpha}), ' \
                         f'Best acc: {self.logger.data["best_acc"]:.2f},' \
                         f'Best params: {self.logger.data["best_params"]} ' \
                         f'Elapse time: {prediction_time:.2f}s'
            bar.next()
        bar.finish()
        Bar.suffix = f'%(index)d/%(max)d'

        message = f'Best acc on validation : {self.logger.data["best_acc"]:.2f}, ' \
                  f'Best params: {self.logger.data["best_params"]}'
        print(message)

        return self.logger.data["best_params"]


    def evaluate(self, best_params):
        cdef:
            KNNClassifier knn

        best_k, best_alpha = best_params
        self.coordinator.edit_cost.update_alpha(best_alpha)

        num_cores = self.parameters.num_cores
        parallel = num_cores > 0

        graphs_train, labels_train = self.coordinator.train_split()
        # graphs_test, labels_test = self.coordinator.val_split()
        graphs_test, labels_test = self.coordinator.test_split()

        knn = KNNClassifier(self.coordinator.ged, parallel, verbose=False)
        knn.train(graphs_train=graphs_train, labels_train=labels_train)

        start_time = time()
        predictions = knn.predict(graphs_test, k=best_k, num_cores=num_cores)
        prediction_time = time() - start_time

        acc = calc_accuracy(np.array(labels_test, dtype=np.int32), predictions)

        message = f'Best acc on Test : {acc:.2f}, best params: {best_params}, time: {prediction_time:.2f}s\n'

        print(message)

        self.logger.data['acc_test'] = acc
        self.logger.data['prediction_time_test'] = prediction_time

        self.logger.save_data()

        current_folder = self.parameters.current_coordinator["folder_dataset"].split('/')[-3]
        filename = f'predictions_{current_folder}.npy'
        if self.parameters.save_predictions:
            self.save_predictions(predictions,
                                  np.array(labels_test, dtype=np.int32),
                                  filename)


cpdef void run_knn_gnn_embedding(parameters, logger):
    run_h_knn_ = RunnerKNNGNN(parameters, logger)
    run_h_knn_.run()
