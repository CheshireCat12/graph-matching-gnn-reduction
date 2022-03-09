import os.path
from argparse import ArgumentParser, Namespace
from glob import glob

from graph_pkg_core.utils.logger import Logger
from kmeans.constants.dataset_constants import DATASETS
from kmeans.run_median_graph import run_median_graph


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


def main(args):
    coordinator = DATASETS[args.dataset]['coordinator']
    dataset_folders = os.path.join(coordinator['folder_dataset'],
                                   '*')
    coordinators = []

    for dataset_folder in glob(dataset_folders):
        if dataset_folder.split('/')[-1] == args.specific_seed:
            coordinator_tmp = dict(coordinator)
            coordinator_tmp['folder_dataset'] = dataset_folder
            coordinator_tmp['folder_labels'] = dataset_folder
            coordinators.append(coordinator_tmp)
        elif not args.specific_seed:
            coordinator_tmp = dict(coordinator)
            coordinator_tmp['folder_dataset'] = dataset_folder + '/data/'
            coordinator_tmp['folder_labels'] = dataset_folder + '/data/'
            coordinators.append(coordinator_tmp)

    if not coordinators:
        raise FileNotFoundError(f'Folder {args.specific_seed} not found!')

    folder_results = f'./results/{args.dataset}/{args.name_experiment}/'

    # Merge the command line parameters with the constant parameters
    arguments = Namespace(**vars(args),
                          **{'coordinators': coordinators,
                             'current_coordinator': None,
                             'folder_results': folder_results})

    filename = os.path.join(folder_results, 'results_general.json')
    logger = Logger(filename)
    logger.data['parameters'] = vars(arguments)


    for idx, coordinator in enumerate(arguments.coordinators):
        current_exp = f'exp_{coordinator["folder_dataset"].split("/")[-3]}'
        print(f'[{1+idx}/{len(arguments.coordinators)}] Run dataset: {current_exp}')
        logger.set_lvl(current_exp)
        arguments.current_coordinator = coordinator

        run_median_graph(arguments, logger)



if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')

    # Parameters of the experiment
    parser.add_argument('--optimize',
                        action='store_true',
                        help='Perform the optimization of the hyperparameters.')
    parser.add_argument('--num-cores', type=int, default=0,
                        help='Run the code in parallel if the num cores > 0.')

    # Dataset parameters
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=DATASETS,
                        help='Choose the dataset.')
    parser.add_argument('--data-folder',
                        type=str,
                        # required=True,
                        help='Choose the data folder of graphs.')
    parser.add_argument('--specific-seed',
                        type=str,
                        default=False,
                        help='Run for a single dataset split.')

    parser.add_argument('--name-experiment', type=str, required=True,
                        help='Specify the experiment name under which to save the experiment')

    parser.add_argument('--alpha',
                        type=float,
                        help='Hyperparameter alpha that weights the strength'
                              'between the cost of the nodes or the edges')

    # parser.add_argument('--test-k', type=int, required=True,
    #                     help='change k of kmeans **To delete')
    # parser.add_argument('--test-seed', type=int, required=True,
    #                     help='change seed of kmeans **To delete')
    parser.add_argument('--range-kmeans', type=int, required=True,
                        nargs='*',
                        help='gives the range for the kmeans')
    parser.add_argument('--n-seeds', type=int, required=True,
                        help='Number of seed to run with kmeans')
    args = parser.parse_args()

    main(args)