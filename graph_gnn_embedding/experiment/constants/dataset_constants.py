DATASETS = {
    'enzymes': {
        'coordinator': {
            'dataset_name': 'enzymes',
            'params_edit_cost': (1.0, 1.0, 1.0, 1.0, 'euclidean'),
            'folder_dataset': './data/ENZYMES/',
            'folder_labels': './data/ENZYMES/'
        }
    },
    'NCI1': {
        'coordinator': {
            'dataset_name': 'NCI1',
            'params_edit_cost': (1.0, 1.0, 1.0, 1.0, 'euclidean'),
            'folder_dataset': './data/NCI1/',
            'folder_labels': './data/NCI1/',
        }
    },
    'mutagenicity': {
        'coordinator': {
            'dataset_name': 'mutagenicity',
            'params_edit_cost': (1.0, 1.0, 1.0, 1.0, 'euclidean'),
            'folder_dataset': './data/Mutagenicity/',
            'folder_labels': './data/Mutagencity/',
        }
    },
    'proteins': {
        'coordinator': {
            'dataset_name': 'proteins',
            'params_edit_cost': (1.0, 1.0, 1.0, 1.0, 'euclidean'),
            # 'folder_dataset': './data/PROTEINS/',
            # 'folder_labels': './data/PROTEINS/',
            'folder_dataset': '../graph-reduction-tripletLoss/results/PROTEINS/',
            'folder_labels': '../graph-reduction-tripletLoss/results/PROTEINS/',
        }
    },
    'dd': {
        'coordinator': {
            'dataset_name': 'dd',
            'params_edit_cost': (1.0, 1.0, 1.0, 1.0, 'euclidean'),
            'folder_dataset': './data/DD/',
            'folder_labels': './data/DD/',
        }
    },
}
