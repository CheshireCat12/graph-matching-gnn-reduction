cdef:
    #########################
    ##  General Constants  ##
    #########################

    ### File Extensions ###
    str EXTENSION_GRAPHML = '*.graphml'
    str EXTENSION_SPLITS = '.cxl'

    #########################
    ##  Folder Constants   ##
    #########################

    dict DEFAULT_FOLDERS_GNN_EMBEDDING = {
        'enzymes': './data_gnn/reduced_graphs_ENZYMES/data/'
    }

    dict DEFAULT_FOLDERS_GNN_EMBEDDING_LABELS = DEFAULT_FOLDERS_GNN_EMBEDDING
