from argparse import ArgumentParser
from glob import glob
from os.path import join
from pathlib import Path
from typing import Dict, Tuple

from knn.run_knn import run_knn

from graph_pkg_core.utils import logger
from knn.constants.dataset_constants import EDIT_COSTS


def init_default_coordinator(dataset: str,
                             edit_costs: Tuple,
                             folder_data: str,
                             **kwargs) -> Dict:
    """
    Initialize the default coordinator

    :param dataset:
    :param edit_costs:
    :param folder_data:
    :param kwargs:
    :return:
    """
    coordinator = {
        'dataset_name': dataset.lower(),
        'params_edit_cost': edit_costs,
        'folder_dataset': folder_data,
        'folder_labels': folder_data,
    }

    return coordinator


def init_seed_coordinators(folder_data: str,
                           specific_seed: str,
                           coordinator: Dict) -> Dict:
    """
    Find all the folders with different seeds in the given folder
    Create a coordinator dict for each of those seeds

    If a specific seed is given only the corresponding coordinator is kept
    otherwise all the cooridinators are returned

    :param folder_data:
    :param coordinator:
    :return:
    """
    # Find all the folders for all the seeds
    seed_folders = join(folder_data, '*')

    coordinators = {}

    for s_folder in glob(seed_folders):
        seed = s_folder.split('/')[-1]
        coordinator_tmp = dict(coordinator)
        coordinator_tmp['folder_dataset'] = s_folder
        coordinator_tmp['folder_labels'] = s_folder
        coordinators[seed] = coordinator_tmp

    seeds = set(coordinators.keys())
    intersection_seeds = seeds.intersection([specific_seed])
    if intersection_seeds:
        selected_seeds = intersection_seeds
    else:
        selected_seeds = coordinators.keys()

    selected_coordinators = {key: val
                             for key, val in coordinators.items()
                             if key in selected_seeds}

    return selected_coordinators


def init_folder_results(folder_results: str,
                        seed: str,
                        **kwargs) -> str:
    """
    Create the folder for the results

    :return:
    """
    complete_folder_results = join(folder_results, seed)
    Path(complete_folder_results).mkdir(parents=True, exist_ok=True)

    return complete_folder_results


def main(args):
    default_coordinator = init_default_coordinator(**vars(args),
                                                   edit_costs=EDIT_COSTS)

    seeded_coordinators = init_seed_coordinators(args.folder_data,
                                                 args.specific_seed,
                                                 default_coordinator)

    # Check if the given folder contain the graphs
    if not seeded_coordinators:
        raise FileNotFoundError(
            f'Graphs in folder {args.folder_data} not found!')
    if args.specific_seed and not seeded_coordinators[args.specific_seed]:
        raise FileNotFoundError(
            f'Graphs in folder {join(args.folder_data, args.specific_seed)} not found!')

    tmp_folder_results = args.folder_results

    filename_results = 'results_general.json'
    logs = logger.Logger(join(tmp_folder_results,
                              filename_results))
    logs.data['parameters'] = vars(args)

    for idx, (seed, coordinator) in enumerate(seeded_coordinators.items()):
        print(f'[{1 + idx}/{len(seeded_coordinators)}] Run dataset: {args.dataset}')
        args.coordinator = coordinator
        args.folder_results = init_folder_results(tmp_folder_results,
                                                  seed)
        logs.set_lvl(seed)
        run_knn(args, logs)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')

    # Parameters of the experiment
    parser.add_argument('--optimize',
                        action='store_true',
                        help='Perform the optimization of the hyperparameters.')
    parser.add_argument('--num-cores', type=int, default=0,
                        help='Run the code in parallel if the num cores > 0.')

    # Hyperparameters to tune
    parser.add_argument('--alpha', nargs='*',
                        help='Choose the alpha parameters that weights the influence of node/edge cost in GED.\n'
                             '(e.g., --alpha start, end, step_size)')
    parser.add_argument('--ks', nargs='*', type=int,
                        help='Choose the parameters k that corresponds to the number of neighbors for the KNN.')

    parser.add_argument('--best-parameters', nargs='*',
                        help='if optimize is False '
                             'then the hyperparameters alpha, k are set based on these values.\n'
                             '(e.g., --best-parameters k alpha)')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--folder-data', type=str, required=True,
                        help='(e.g. ../../data/proteins/100/ )')
    parser.add_argument('--folder-results', type=str, required=True,
                        help='(e.g. --folder-results ./results/ )')
    parser.add_argument('--specific-seed', type=str, default=False)

    # Value to save
    parser.add_argument('--save-predictions',
                        action='store_true',
                        help='Save the predictions done on the test set.')
    parser.add_argument('--save-dist-matrix',
                        action='store_true',
                        help='Save the distance matrix between the graphs in the train and test set.')

    args = parser.parse_args()

    main(args)
