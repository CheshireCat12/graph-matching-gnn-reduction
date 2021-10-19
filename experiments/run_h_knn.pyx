from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier
from graph_pkg.utils.constants cimport PERCENT_HIERARCHY
from graph_pkg.algorithm.knn cimport KNNClassifier
from hierarchical_graph.hierarchical_graphs cimport HierarchicalGraphs
from hierarchical_graph.centrality_measure.pagerank import PageRank
from hierarchical_graph.centrality_measure.betweenness import Betweenness
from experiments.runner import Runner
from hierarchical_graph.gatherer_hierarchical_graphs cimport GathererHierarchicalGraphs as GAG

from graph_pkg.utils.functions.helper import calc_accuracy, calc_f1
from pathlib import Path
import numpy as np
cimport numpy as np
from time import time
import os
from itertools import product
from collections import defaultdict
import pandas as pd
from progress.bar import Bar

# 1. Run the experiment with the given dataset 2 times (Pagerank, Betweenness)
#    1.1 with the optimize: True and percentage_to_opt: [1.,0, ....]
#    1.2 Find the best parameters for percentage (lambda): 1.0 (full graphs)
#    1.3 Report those parameters in the configuration file
# 2. Run the experiment for the given dataset 2 times (PageRank, Betweenness)
#    2.1 This time with optimize: False and check_all_percentages: True
#    2.2 It will take the best params from the full graphs and do the comparison for all the percentages without optimizing.
#


class RunnerHKnn(Runner):

    def __init__(self, parameters):
        super(RunnerHKnn, self).__init__(parameters)

    def run(self):
        print('Run KNN with Reduced Graphs')

        # Init the graph gatherer
        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.current_centrality_measure
        percentages = self.parameters.hierarchy_params['percentages']

        run_full_dataset = False if self.parameters.coordinator['dataset'] in ['collab', 'reddit_binary'] else True

        self.save_stats('The code is running\n', 'log.txt', save_params=False)

        self.gag = GAG(coordinator_params, percentages,
                       centrality_measure, activate_aggregation=False,
                       full_dataset=run_full_dataset, verbose=True)

        self.save_stats('The graphs are loaded and reduced\n', 'log.txt')

        self.test_evaluation = []

        if self.parameters.optimize:
            for percentage_to_opt in self.parameters.percentages_to_opt:
                self.parameters.current_percentage_to_opt = percentage_to_opt
                best_params = self.optimization()

                self.evaluate(best_params)
        elif self.parameters.current_centrality_measure == 'random':

            best_params = tuple(self.parameters.h_knn[1.0].values())

            # percentages_to_check = [1.0, 0.8, 0.6, 0.4, 0.2]
            percentages_to_check = self.parameters.hierarchy_params['percentages'] #[1.0]
            np.random.seed(42)
            seeds = np.random.randint(1000, size=self.parameters.n_random_turns)

            random_acc = []
            random_time = []

            for lambda_ in percentages_to_check:
                self.test_evaluation = []
                for seed in seeds:
                    self.gag = GAG(coordinator_params, percentages,
                                   centrality_measure, activate_aggregation=False,
                                   full_dataset=run_full_dataset, verbose=True, new_seed=seed)
                    self.parameters.current_percentage_to_opt = lambda_
                    self.evaluate(best_params)

                print(self.test_evaluation)
                random_acc.append([acc for _, acc, _ in self.test_evaluation])
                random_time.append([time for _, _, time in self.test_evaluation])

            Path(self.parameters.folder_results).mkdir(parents=True, exist_ok=True)
            filename_values = os.path.join(self.parameters.folder_results, f'random_values.csv')
            filename_time = os.path.join(self.parameters.folder_results, f'time.csv')

            dataframe = pd.DataFrame(np.array(random_acc).T, index=seeds, columns=percentages_to_check)
            dataframe.to_csv(filename_values)


            dataframe = pd.DataFrame(np.array(random_time).T, index=seeds, columns=percentages_to_check)
            dataframe.to_csv(filename_time)

        else:
            params_from_file = self.parameters.h_knn

            # Retrieve the parameters from the full graph optimization
            current_percentage = params_from_file['current_percentage']
            best_params = tuple(params_from_file[current_percentage].values())

            # Select on which percentage test the obtained parameters
            if self.parameters.check_all_percentages:
                percentage_to_check = self.parameters.hierarchy_params['percentages']
            else:
                percentage_to_check = [current_percentage]

            for lambda_ in percentage_to_check:
                # temp_edit_cost = self.parameters.coordinator['params_edit_cost']
                self.parameters.current_percentage_to_opt = lambda_

                self.evaluate(best_params)
                # self.parameters.coordinator['params_edit_cost'] = temp_edit_cost

        # if self.parameters.optimize:
        #     self.save_stats(message, f'{centrality_measure}_test_results_h_knn_{current_percentage_opt}.txt')
        # else:
        #     write_params = current_percentage_opt == 1.0
        #     self.save_stats(message, f'{centrality_measure}_test_results_not_opt.txt', save_params=write_params)
        #
        #
        # Path(self.parameters.folder_results).mkdir(parents=True, exist_ok=True)
        # filename = os.path.join(self.parameters.folder_results,
        #                         f'{centrality_measure}_fine_tuning_{current_percentage_to_opt}.csv')
        #
        # dataframe = pd.DataFrame(accuracies, index=alphas, columns=ks)
        # dataframe.to_csv(filename)

    def optimization(self):
        cdef:
            KNNClassifier knn

        ##################
        # Set parameters #
        ##################

        centrality_measure = self.parameters.current_centrality_measure
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        current_percentage_to_opt = self.parameters.current_percentage_to_opt

        knn = KNNClassifier(self.gag.coordinator.ged, parallel, verbose=False)
        knn.train(self.gag.h_graphs_train.hierarchy[current_percentage_to_opt],
                  self.gag.labels_train)

        # Hyperparameters to tune
        alpha_start, alpha_end, alpha_step = self.parameters.tuning['alpha']
        alphas = [alpha_step * i for i in range(alpha_start, alpha_end)]
        ks = self.parameters.tuning['ks']

        best_acc = float('-inf')
        best_params = (None, None)
        accuracies = defaultdict(list)

        hyperparameters = product(ks, alphas)
        len_hyperparameters = len(ks) * len(alphas)

        bar = Bar(f'Processing Graphs level: {current_percentage_to_opt}', max=len_hyperparameters)

        for k_param, alpha in hyperparameters:
            alpha = round(alpha, 2)
            self.gag.coordinator.edit_cost.update_alpha(alpha)

            predictions = knn.predict(self.gag.h_graphs_val.hierarchy[current_percentage_to_opt],
                                      k=k_param, num_cores=num_cores)

            acc = calc_accuracy(np.array(self.gag.labels_val, dtype=np.int32), predictions)

            if acc >= best_acc:
                best_acc = acc
                best_params = (k_param, alpha)

            accuracies[k_param].append(acc)

            Bar.suffix = f'%(index)d/%(max)d | Best acc {best_acc:.2f}, with {best_params}'
            bar.next()
        bar.finish()

        # Save the validation accuracy per hyperparameter
        Path(self.parameters.folder_results).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.parameters.folder_results,
                                f'{centrality_measure}_fine_tuning_{current_percentage_to_opt}.csv')

        dataframe = pd.DataFrame(accuracies, index=alphas, columns=ks)
        dataframe.to_csv(filename)

        # Save the best acc on validation
        message = f'Best acc on validation {current_percentage_to_opt}: {best_acc:.2f}, best params: {best_params}'
        print(message)
        self.save_stats(message, f'{centrality_measure}_opt_h_knn.txt',
                        save_params=False)

        return best_params


    def evaluate(self, best_params):
        cdef:
            KNNClassifier knn

        best_k, best_alpha = best_params
        print(best_params)
        self.gag.coordinator.edit_cost.update_alpha(best_alpha)
        # params_edit_cost = self.parameters.coordinator['params_edit_cost']
        # self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)

        centrality_measure = self.parameters.current_centrality_measure
        current_percentage_opt = self.parameters.current_percentage_to_opt
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel

        knn = KNNClassifier(self.gag.coordinator.ged, parallel, verbose=False)
        knn.train(self.gag.h_graphs_train.hierarchy[current_percentage_opt],
                  self.gag.labels_train)

        start_time = time()
        predictions = knn.predict(self.gag.h_graphs_test.hierarchy[current_percentage_opt],
                                  k=best_k, num_cores=num_cores)
        prediction_time = time() - start_time

        acc = calc_accuracy(np.array(self.gag.labels_test, dtype=np.int32), predictions)

        message = f'Best acc on Test {current_percentage_opt}: {acc:.2f}, best params: {best_params}, time: {prediction_time}\n'
        print(message)



        if self.parameters.optimize:
            self.save_stats(message, f'{centrality_measure}_test_results_h_knn_{current_percentage_opt}.txt')
        elif self.parameters.current_centrality_measure == 'random':
            pass
        else:
            write_params = current_percentage_opt == 1.0
            self.save_stats(message, f'{centrality_measure}_test_results_not_opt.txt', save_params=write_params)

        self.test_evaluation.append((current_percentage_opt, acc, prediction_time))

        if self.parameters.save_dist_matrix:
            # Save the validation accuracy per hyperparameter
            folder = os.path.join(self.parameters.folder_results, 'distances')
            Path(folder).mkdir(parents=True, exist_ok=True)
            filename = os.path.join(folder,
                                    f'{centrality_measure}_dist_{current_percentage_opt}.npy')

            with open(filename, 'wb') as f:
                np.save(f, knn.current_distances)

        filename = f'prediction_full'
        self.save_predictions(predictions, np.array(self.gag.labels_test, dtype=np.int32), f'{filename}.npy')
        # Reinitialize the coordinator params
        # self.parameters.coordinator['params_edit_cost'] = params_edit_cost

class HyperparametersTuning:

    __MEASURES = {
        'pagerank': PageRank(),
        'betweenness': Betweenness(),
    }

    def __init__(self, parameters):
        self.parameters = parameters

    # def fine_tune(self):
    #     print('Finetune the parameters')
    #
    #     # set parameters to tune
    #     alpha_start, alpha_end, alpha_step = self.parameters.tuning['alpha']
    #     alphas = [alpha_step * i for i in range(alpha_start, alpha_end)]
    #     ks = self.parameters.tuning['ks']
    #
    #     params_edit_cost = self.parameters.coordinator['params_edit_cost']
    #
    #     accuracies = defaultdict(list)
    #
    #     best_acc = float('-inf')
    #     best_params = (None,)
    #
    #     for k, alpha in product(ks, alphas):
    #         print('+ Tuning parameters +')
    #         print(f'+ alpha: {alpha:.2f}, k: {k} +\n')
    #
    #         self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, alpha)
    #         self.parameters.k = k
    #
    #         acc, _ = self._run_pred_val_test()
    #         accuracies[k].append(acc)
    #
    #         if acc > best_acc:
    #             best_acc = acc
    #             best_params = (alpha, k)
    #
    #         # break
    #
    #     dataframe = pd.DataFrame(accuracies, index=alphas, columns=ks)
    #     print(dataframe)
    #
    #     print(f'Best acc on validation: {best_acc}, best params: {best_params}')
    #
    #     Path(self.parameters.folder_results).mkdir(parents=True, exist_ok=True)
    #     filename = os.path.join(self.parameters.folder_results, 'fine_tuning_heuristic.csv')
    #     dataframe.to_csv(filename)
    #
    #     best_alpha, best_k = best_params
    #     self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)
    #     self.parameters.k = best_k
    #     acc_test, time_pred = self._run_pred_val_test(validation=False)
    #     _write_results(acc_test, time_pred, self.parameters, 'accuracy_test_heuristic.txt')
    #
    # def run_hierarchy(self):
    #     print('Run Hierarchy')
    #
    #     # set parameters to tune
    #     percentages = self.parameters.hierarchy_params['percentages']
    #     measures = self.parameters.hierarchy_params['centrality_measures']
    #
    #     params_edit_cost = self.parameters.coordinator['params_edit_cost']
    #     best_alpha = self.parameters.best_alpha
    #     self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)
    #
    #     for measure in measures:
    #         print('+ Tweaking parameters +')
    #         print(f'+ Measure: {measure} +\n')
    #
    #         self.parameters.centrality_measure = measure
    #
    #         acc, time_pred = self._run_pred_val_test(validation=False)
    #
    #         filename = f'refactor_measure_{measure}_heuristic.txt'
    #
    #         _write_results_new(acc, time_pred, self.parameters, filename)
    #
    #
    # def _run_pred_val_test(self, validation=True):
    #     cdef:
    #         CoordinatorClassifier coordinator
    #         KNNClassifier knn
    #
    #     # Set parameters
    #     coordinator_params = self.parameters.coordinator
    #     centrality_measure = self.parameters.centrality_measure
    #     deletion_strategy = self.parameters.deletion_strategy
    #     k = self.parameters.k
    #     parallel = self.parameters.parallel
    #
    #     # Retrieve graphs with labels
    #     coordinator = CoordinatorClassifier(**coordinator_params)
    #     graphs_train, labels_train = coordinator.train_split(conv_lbl_to_code=True)
    #     graphs_val, labels_val = coordinator.val_split(conv_lbl_to_code=True)
    #     graphs_test, labels_test = coordinator.test_split(conv_lbl_to_code=True)
    #
    #     # Set the graph hierarchical
    #     measure = self.__MEASURES[centrality_measure]
    #     h_graphs_train = HierarchicalGraphs(graphs_train, measure)
    #     h_graphs_val = HierarchicalGraphs(graphs_val, measure)
    #     h_graphs_test = HierarchicalGraphs(graphs_test, measure)
    #
    #
    #     accuracies, times = [], []
    #     for percentage in self.parameters.hierarchy_params['percentages']:
    #         # Create and train the classifier
    #         knn = KNNClassifier(coordinator.ged, parallel)
    #         knn.train(h_graphs_train.hierarchy[percentage], labels_train)
    #
    #         acc, time_pred, predictions, np_lbl_test = _do_prediction(knn,
    #                                                                   h_graphs_test.hierarchy[percentage],
    #                                                                   labels_test, k, 'Test')
    #
    #         if self.parameters.save_predictions:
    #             Path(self.parameters.folder_results).mkdir(parents=True, exist_ok=True)
    #
    #             name = f'prediction_{percentage*100:.0f}_{self.parameters.centrality_measure}.npy'
    #             filename = os.path.join(self.parameters.folder_results, name)
    #             with open(filename, 'wb') as f:
    #                 np.save(f, np_lbl_test)
    #                 np.save(f, predictions)
    #
    #         accuracies.append(acc)
    #         times.append(time_pred)
    #
    #     return accuracies, times
        #
        # # Create the reduced graphs
        # graphs_train_reduced = h_graph.create_hierarchy_percent(graphs_train,
        #                                                         percentage,
        #                                                         deletion_strategy,
        #                                                         verbose=True)
        # if validation:
        #     graphs_val_reduced = h_graph.create_hierarchy_percent(graphs_val,
        #                                                       percentage,
        #                                                       deletion_strategy,
        #                                                       verbose=True)
        # else:
        #     graphs_test_reduced = h_graph.create_hierarchy_percent(graphs_test,
        #                                                        percentage,
        #                                                        deletion_strategy,
        #                                                        verbose=True)
        # Perform prediction
        # if validation:
        #     acc, time_pred = _do_prediction(knn, graphs_val_reduced, labels_val, k, 'Validation')
        # else:
        #     acc, time_pred = _do_prediction(knn, graphs_test_reduced, labels_test, k, 'Test')

        # acc_test, time_test = _do_prediction(knn, graphs_test_reduced, labels_test, k, 'Test')
        #
        # _write_results(acc_val, time_val, self.parameters)
        # _write_results(acc_test, time_test, self.parameters)


cpdef void run_h_knn(parameters):
    run_h_knn_ = RunnerHKnn(parameters)
    run_h_knn_.run()
