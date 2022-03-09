import json
import numpy as np
cimport numpy as np

from progress.bar import Bar
from os.path import join

from graph_pkg_core.coordinator.coordinator_vector_classifier cimport CoordinatorVectorClassifier
from graph_pkg_core.graph.graph cimport Graph

cpdef int[::1] connected_components(Graph graph):
    """
    Use BFS to compute the compute the component in the given graph
    :param adj: 
    :return: 
    """
    cdef:
        int cur_vertex
        int cur_component = 0
        list queue = []
        int[:, ::1] adj

    adj = graph.adjacency_matrix
    components = np.zeros(adj.shape[0], dtype=np.int32)

    for idx, _ in enumerate(adj):
        # Check if the current vertex has already been visited
        if components[idx] != 0:
            continue
        cur_component += 1
        queue.append(idx)

        while queue:
            cur_vertex = queue.pop()
            cur_neighbors = adj[cur_vertex]
            components[cur_vertex] = cur_component

            # Find the neighbors of the current node that are not yet visited
            available_neighbors = np.where((np.asarray(cur_neighbors)==1) &
                                           (components == 0))[0]

            components[available_neighbors] = cur_component
            queue += available_neighbors.tolist()

    return components

cpdef int num_connected_components(Graph graph):
    """
    Compute the number of connected components
    :param adj: 
    :return: 
    """
    components = np.asarray(connected_components(graph))

    return np.unique(components).size

cpdef int max_degree(Graph graph):
    return np.max(graph.degrees())

cpdef int[::1] isolated_nodes(Graph graph):
    """
    Retrieve the list of isolated nodes
    A node is isolated if its node degree is 0
    :param graph: 
    :return: 
    """
    degrees = np.asarray(graph.degrees())
    # print(np.where(degrees == 0)[0])
    return np.where(degrees == 0)[0].astype(np.int32)

cpdef int num_isolated_nodes(Graph graph):
    return np.asarray(isolated_nodes(graph)).size

cpdef int num_edges(Graph graph):
    return np.sum(graph.degrees()) // 2

cpdef void run_graph_analysis(arguments):
    cdef:
        CoordinatorVectorClassifier coordinator

    coordinator_params = arguments.current_coordinator
    coordinator = CoordinatorVectorClassifier(**coordinator_params)

    tr_data, tr_labels = coordinator.train_split()

    graph_stats = {
        'n_connected_components': [],
        'n_isolated_nodes': [],
        'n_nodes': [],
        'n_edges': [],
        'mean_degrees': [],
        'max_degrees': [],
    }

    with Bar('Processing', max=len(tr_data)) as bar:
        for graph in tr_data:
            graph_stats['n_connected_components'].append(num_connected_components(graph))
            graph_stats['n_isolated_nodes'].append(num_isolated_nodes(graph))
            graph_stats['n_nodes'].append(graph.num_nodes_max)
            graph_stats['n_edges'].append(num_edges(graph))
            graph_stats['mean_degrees'].append(np.mean(graph.degrees()))
            graph_stats['max_degrees'].append(int(np.max(graph.degrees())))

            # visualize(graph, arguments.folder_results)

    print(graph_stats)
    filename_graph_stats = join(arguments.folder_results,
                                f'graph_stats.json')
    with open(filename_graph_stats, 'w') as file:
        json.dump(graph_stats, file)
