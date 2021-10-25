import pytest

from graph_gnn_embedding.loader_gnn_embedding.loader_gnn_embedding_base import LoaderGNNEmbeddingBase


############## Base ##############

@pytest.mark.parametrize('folder, num_graphs',
                         [('./data/reduced_graphs_ENZYMES/data/', 600)])
def test_base_loader_embedding(folder, num_graphs):
    loader_base = LoaderGNNEmbeddingBase(folder)
    graphs = loader_base.load()

    print(graphs[0])
    # assert False
    assert len(graphs) == num_graphs
