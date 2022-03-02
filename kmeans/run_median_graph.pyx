import csv
import random
from os.path import join
from pathlib import Path
from time import time

import numpy as np
cimport numpy as np
from collections import defaultdict
from itertools import product
from progress.bar import Bar

from graph_pkg_core.experiment.runner import Runner
from graph_pkg_core.algorithm.kmeans cimport Kmeans
from graph_pkg_core.coordinator.coordinator_vector_classifier cimport CoordinatorVectorClassifier


class RunnerMedian(Runner):

    def __init__(self, parameters, logger):
        super(RunnerMedian, self).__init__(parameters)
        self.logger = logger

    def run(self) -> None:
        print('Run Median Graph')

        # Init the graph gatherer
        coordinator_params = self.parameters.current_coordinator

        self.coordinator = CoordinatorVectorClassifier(**coordinator_params)

        best_alpha = int(self.parameters.alpha)

        self.coordinator.edit_cost.update_alpha(best_alpha)

        self.find_kmeans()

    def _format_name(self, name: str) -> str:
        return name

    def _write_csv(self, path, fields, data):
        with open(path, mode='w') as f:
            csv_writer = csv.writer(f, delimiter=',')

            csv_writer.writerow(fields)
            csv_writer.writerows(data)

    def save_centroids(self,
                       list graphs_per_cls,
                       Kmeans kmeans,
                       int cls_,
                       int n_cluster) -> None:
        """
        Save the graph index of the centroids and the corresponding centroid
        for each graph.
        """
        seed_split = self.coordinator.folder_dataset.split('/')[-1]

        filename_centroids = f'centroids'\
                             f'_seed_split_{seed_split}'\
                             f'_n_cluster_{n_cluster}'\
                             f'_cls_{cls_}.csv'

        filename_correspondance = f'centroids_correspondance'\
                                  f'_seed_split_{seed_split}'\
                                  f'_n_cluster_{n_cluster}'\
                                  f'_cls_{cls_}.csv'

        folder_centroids = join(self.folder, 'centroids')
        Path(folder_centroids).mkdir(parents=True, exist_ok=True)
        path_centroids = join(folder_centroids,
                                      filename_centroids)
        path_correspondence = join(folder_centroids,
                                      filename_correspondance)

        idx_centroids = [self._format_name(centroid.name)
                         for centroid in kmeans.centroids]

        print('name and length')
        print([centroid.name for centroid in kmeans.centroids])
        print([len(centroid) for centroid in kmeans.centroids])

        correspondance = [(self._format_name(graph.name),
                           self._format_name(kmeans.centroids[centroid].name))
                          for graph, centroid in zip(graphs_per_cls, kmeans.labels)]

        fields_centroids = ['idx_centroid']
        fields_correspondance = ['idx_graph', 'idx_centroid']

        self._write_csv(path_centroids,
                        fields_centroids,
                        idx_centroids)
        self._write_csv(path_correspondence,
                        fields_correspondance,
                        correspondance)

    def find_kmeans(self):
        n_cores = self.parameters.num_cores
        parallel = n_cores > 0

        graphs_train, labels_train = self.coordinator.train_split()

        labels = np.array([int(val) for val in labels_train], dtype=np.int32)
        individual_class = set(labels)

        random.seed(42)
        seeds = [random.randint(1, 10000) for _ in range(self.parameters.n_seeds)]
        n_clusters = range(*[int(val) for val in self.parameters.range_kmeans])

        for idx, cls_ in enumerate(individual_class): # sorted(individual_class, reverse=True):
            self.logger.data[f'cls_{cls_}'] = {
                'err_and_seed_per_k': defaultdict(lambda : list()),
                'best_err_per_k': defaultdict(lambda : float('inf')),
                'best_seed_per_k': {},
                'centroids': {}
            }

            cls_idx = np.where(np.array(labels) == cls_)[0]
            graphs_per_cls = [graphs_train[idx] for idx in cls_idx]

            bar = Bar(f'Processing clustering [{idx+1}/{len(individual_class)}]',
                      max=len(n_clusters) * len(seeds))
            for n_cluster, new_seed in product(n_clusters, seeds):
                kmeans = Kmeans(n_cluster,
                                self.coordinator.ged,
                                max_iter=30,
                                seed=new_seed,
                                n_cores=n_cores)

                kmeans.fit(graphs_per_cls)
                err_clustering = kmeans.error
                print(f'Error: {err_clustering}')

                self.logger.data[f'cls_{cls_}']['err_and_seed_per_k'][n_cluster].append((err_clustering, new_seed))

                if err_clustering <= self.logger.data[f'cls_{cls_}']['best_err_per_k'][n_cluster]:
                    self.logger.data[f'cls_{cls_}']['best_err_per_k'][n_cluster] = err_clustering
                    self.logger.data[f'cls_{cls_}']['best_seed_per_k'][n_cluster] = new_seed
                    self.logger.data[f'cls_{cls_}']['centroids'][n_cluster] = [centroid.name for centroid in kmeans.centroids]
                    self.save_centroids(graphs_per_cls, kmeans, cls_, n_cluster)

                self.logger.save_data()

                bar.next()
            bar.finish()


cpdef void run_median_graph(parameters, logger):
    run_h_knn_ = RunnerMedian(parameters, logger)
    run_h_knn_.run()
