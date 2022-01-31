import numpy as np
cimport numpy as np
import pandas as pd
from progress.bar import Bar

from graph_pkg.experiment.runner import Runner
from graph_pkg.algorithm.knn cimport KNNClassifier
from graph_gnn_embedding.utils.coordinator.coordinator_gnn_embedding_classifier import CoordinatorGNNEmbeddingClassifier

class RunnerMedian(Runner):

    def __init__(self, parameters, logger):
        super(RunnerMedian, self).__init__(parameters)
        self.logger = logger

    def run(self):
        print('Run KNN with Reduced Graphs')

cpdef void run_knn_gnn_embedding(parameters, logger):
    run_h_knn_ = RunnerMedian(parameters, logger)
    run_h_knn_.run()
