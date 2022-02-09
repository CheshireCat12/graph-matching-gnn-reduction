import os
import random
from time import time

import numpy as np
cimport numpy as np
from collections import defaultdict
from itertools import product
from progress.bar import Bar

from graph_pkg.experiment.runner import Runner
from graph_pkg.algorithm.knn cimport KNNClassifier
from graph_pkg.algorithm.kmeans cimport Kmeans
from graph_gnn_embedding.utils.coordinator.coordinator_gnn_embedding_classifier import CoordinatorGNNEmbeddingClassifier


class RunnerMedian(Runner):

    def __init__(self, parameters, logger):
        super(RunnerMedian, self).__init__(parameters)
        self.logger = logger

    def run(self):
        print('Run Median Graph')

        # Init the graph gatherer
        coordinator_params = self.parameters.current_coordinator

        self.save_stats('The code is running\n', 'log.txt', save_params=False)

        self.coordinator = CoordinatorGNNEmbeddingClassifier(**coordinator_params)

        self.save_stats('The graphs are loaded\n', 'log.txt', save_params=False)

        best_k, best_alpha = [type_(val) for val, type_ in zip(self.parameters.best_parameters, [int, float])]
        self.coordinator.edit_cost.update_alpha(best_alpha)

        self.find_kmeans()

    def save_centroids(self, list centroids, int n_cluster):
        seed_split = self.coordinator.folder_dataset.split('/')[-3]
        name = f'centroids_seed_split_{seed_split}_n_cluster_{n_cluster}'
        filename = os.path.join(self.folder, name)

        message = ','.join(centroid.name for centroid in centroids)
        print(message)

        # with open(filename, mode='a+') as fp:
        #     fp.write(f'{message}')
        # filename
        #seed_split
        #n_cluster

        # save in the file
        # - seed_init

    def find_kmeans(self):
        n_cores = self.parameters.num_cores
        parallel = n_cores > 0

        graphs_train, labels_train = self.coordinator.train_split()

        labels = np.array([int(val) for val in labels_train], dtype=np.int32)
        individual_class = set(labels)

        random.seed(42)
        seeds = [random.randint(1, 10000) for _ in range(self.parameters.n_seeds)]
        n_clusters = range(*[int(val) for val in self.parameters.range_kmeans])

        for idx, cls in enumerate(individual_class): # sorted(individual_class, reverse=True):
            self.logger.data[f'cls_{cls}'] = {
                'err_and_seed_per_k': defaultdict(lambda : list()),
                'best_err_per_k': defaultdict(lambda : float('inf')),
                'best_seed_per_k': {},
                'centroids': {}
            }

            cls_idx = np.where(np.array(labels) == cls)[0]
            graphs_per_cls = [graphs_train[idx] for idx in cls_idx]

            bar = Bar(f'Processing clustering [{idx+1}/{len(individual_class)}]', max=len(n_clusters) * len(seeds))
            for n_cluster, new_seed in product(n_clusters, seeds):
                kmeans = Kmeans(n_cluster,
                                self.coordinator.ged,
                                max_iter=30,
                                seed=new_seed,
                                n_cores=n_cores)

                kmeans.fit(graphs_per_cls)
                err_clustering = kmeans.error

                self.logger.data[f'cls_{cls}']['err_and_seed_per_k'][n_cluster].append((err_clustering, new_seed))

                if err_clustering <= self.logger.data[f'cls_{cls}']['best_err_per_k'][n_cluster]:
                    self.logger.data[f'cls_{cls}']['best_err_per_k'][n_cluster] = err_clustering
                    self.logger.data[f'cls_{cls}']['best_seed_per_k'][n_cluster] = new_seed
                    self.logger.data[f'cls_{cls}']['centroids'][n_cluster] = [centroid.name for centroid in kmeans.centroids]
                    # self.save_centroids(kmeans.centroids, n_cluster)

                self.logger.save_data()
                bar.next()
            bar.finish()
            # break


    def find_median(self, k=1):
        cdef:
            int[::1] labels
            double[:, ::1] distances

        num_cores = self.parameters.num_cores
        parallel = num_cores > 0

        graphs_train, labels_train = self.coordinator.train_split()

        labels = np.array([int(val) for val in labels_train], dtype=np.int32)
        individual_class = set(labels)

        class_split = dict()

        knn = KNNClassifier(self.coordinator.ged, parallel, verbose=False)

        start = time()
        for class_ in individual_class:
            class_indices = np.where(np.array(labels)==class_)[0]
            print(class_indices)
            graphs_per_cls = [graphs_train[idx] for idx in class_indices]


            class_split[class_] = [graphs_per_cls, class_indices]
            print(len(graphs_per_cls))

            knn.train(graphs_train=graphs_per_cls, labels_train=labels_train)

            distances = knn.compute_dist(graphs_per_cls, num_cores=num_cores)

            print(np.sum(distances, axis=0))
            idx_centroid = np.argmin(np.sum(distances, axis=1))

            print(graphs_per_cls[idx_centroid])

        print(f'time: {time() - start:.2f}')

cpdef void run_median_graph(parameters, logger):
    run_h_knn_ = RunnerMedian(parameters, logger)
    run_h_knn_.run()
