import os.path
from argparse import ArgumentParser, Namespace
from glob import glob
from bunch import Bunch

from graph_pkg.utils.functions.load_config import load_config
import graph_pkg.utils.logger as logger
from graph_gnn_embedding.experiment.constants.dataset_constants import DATASETS
from graph_gnn_embedding.experiment.run_knn_gnn_embedding import run_knn_gnn_embedding

__EXPERIMENTS_GNN = {
    'knn': run_knn_gnn_embedding
}

__DATASETS = [
    'enzymes',
]


def print_fancy_title(text, size_max=50):
    """
    Print the title in a fancy manner :)

    :param text:
    :param size_max:
    :return:
    """
    border = (size_max - len(text) - 4) // 2
    is_odd = len(text) % 2 != 0
    print(f'\n{"=" * size_max}\n'
          f'=={" " * border}{text}{" " * (border + is_odd)}==\n'
          f'{"=" * size_max}')


# def run_experiment(args):
#         for dataset in __DATASETS:
#             args.dataset = dataset
#
#             _run(args, parameters)
#     else:
#         _run(args, parameters)
#
#     print_fancy_title('Final')


# def _run(args, parameters):
#     # Fusion the selected dataset parameters with the general parameters
#     parameters = Bunch({**parameters[args.dataset], **parameters['general']})
#
#     print_fancy_title('Parameters')
#     print(parameters)
#
#     print_fancy_title('Run')
#     __EXPERIMENTS_GNN[args.exp](parameters)



if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')

    # Parameters of the experiment
    parser.add_argument('--experiment', type=str, required=True,
                        choices=__EXPERIMENTS_GNN.keys(),
                        help='Choose the experiment to run.')
    parser.add_argument('--optimize',
                        action='store_true',
                        help='Perform the optimization of the hyperparameters.')
    parser.add_argument('--num-cores', type=int, default=0,
                        help='Run the code in parallel if the num cores > 0.')

    # Hyperparameters
    parser.add_argument('--alpha', nargs='*',
                        help='Choose the alpha parameters that weights the influence of node/edge cost in GED.\n'
                             '(e.g., --alpha start, end, step_size)')
    parser.add_argument('--ks', nargs='*', type=int,
                        help='Choose the parameters k that corresponds to the number of neighbors for the KNN.')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        choices=__DATASETS,
                        help='Choose the dataset. (default: enzymes)')
    parser.add_argument('--percentage', type=str, required=True,
                        choices=['100', '50'],
                        help='Choose the percentage of reduced graphs.')
    parser.add_argument('--specific-name', type=str, default=False,
                        help='Run for a single dataset split.')

    # Value to save
    parser.add_argument('--save-predictions',
                        action='store_true',
                        help='Save the predictions done on the test set.')
    parser.add_argument('--save-dist-matrix',
                        action='store_true',
                        help='Save the distance matrix between the graphs in the train and test set.')

    parser.add_argument('--name-experiment', type=str, required=True,
                        help='Specify the experiment name under which to save the experiment')

    args = parser.parse_args()

    coordinator = DATASETS[args.dataset]['coordinator']
    dataset_folders = os.path.join(coordinator['folder_dataset'],
                                   args.percentage,
                                   '*')
    coordinators = []

    for dataset_folder in glob(dataset_folders):
        if dataset_folder.split('/')[-1] == args.specific_name:
            coordinator_tmp = dict(coordinator)
            coordinator_tmp['folder_dataset'] = dataset_folder + '/data/'
            coordinator_tmp['folder_labels'] = dataset_folder + '/data/'
            coordinators.append(coordinator_tmp)
        elif not args.specific_name:
            coordinator_tmp = dict(coordinator)
            coordinator_tmp['folder_dataset'] = dataset_folder + '/data/'
            coordinator_tmp['folder_labels'] = dataset_folder + '/data/'
            coordinators.append(coordinator_tmp)

    if not coordinators:
        raise FileNotFoundError(f'Folder {args.specific_name} not found!')

    folder_results = f'./results/{args.experiment}/{args.dataset}/{args.name_experiment}/'

    # Merge the command line parameters with the constant parameters
    arguments = Namespace(**vars(args),
                          **{'coordinators': coordinators,
                             'current_coordinator': None,
                             'folder_results': folder_results})


    filename = os.path.join(folder_results, 'results_general.json')
    logger = logger.Logger(filename)
    logger.data['parameters'] = vars(arguments)

    print(arguments)
    print('##########\n\n')

    for idx, coordinator in enumerate(arguments.coordinators):
        current_exp = f'exp_{coordinator["folder_dataset"].split("/")[-3]}'

        print(current_exp)
        logger.set_lvl(current_exp)
        arguments.current_coordinator = coordinator

        __EXPERIMENTS_GNN[arguments.experiment](arguments, logger)
