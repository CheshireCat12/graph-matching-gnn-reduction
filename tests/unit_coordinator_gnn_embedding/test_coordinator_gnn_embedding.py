import pytest

from graph_gnn_embedding.utils.coordinator_gnn_embedding.coordinator_gnn_embedding import CoordinatorGNNEmbedding

def test_default_enzymes():
    coordinator = CoordinatorGNNEmbedding(dataset='enzymes',
                                          params_edit_cost=(1., 1., 1., 1., 'euclidean'),
                                          folder_dataset='./data/reduced_graphs_ENZYMES/data/')

    assert len(coordinator.graphs) == 600