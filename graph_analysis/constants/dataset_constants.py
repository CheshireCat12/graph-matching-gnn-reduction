EDIT_COSTS = (1.0, 1.0, 1.0, 1.0, 'euclidean')
DATASETS = {
    'enzymes': {
        'coordinator': {
            'dataset_name': 'enzymes',
            'params_edit_cost': (1.0, 1.0, 1.0, 1.0, 'euclidean'),
            'folder_dataset': '../../data/enzymes/',
            'folder_labels': '../../data/enzymes/'
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
            'folder_dataset': '../../data/mutagenicity/',
            'folder_labels': '../../data/mutagencity/',
        }
    },
    'proteins': {
        'coordinator': {
            'dataset_name': 'proteins',
            'params_edit_cost': (1.0, 1.0, 1.0, 1.0, 'euclidean'),
        }
    },
    'dd': {
        'coordinator': {
            'dataset_name': 'dd',
            'params_edit_cost': (1.0, 1.0, 1.0, 1.0, 'euclidean'),
            'folder_dataset': '../../data/dd/',
            'folder_labels': '../../data/dd/',
        }
    },
}
