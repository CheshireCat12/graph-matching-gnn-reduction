from argparse import ArgumentParser, Namespace
from glob import glob
from os.path import join
from pathlib import Path
from typing import Dict, Tuple

from graph_analysis.constants.dataset_constants import EDIT_COSTS
from graph_analysis.run_analysis import run_graph_analysis


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
    default_coordinator = init_default_coordinator(**vars(args), edit_costs=EDIT_COSTS)

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

    for idx, (seed, coordinator) in enumerate(seeded_coordinators.items()):
        print(f'[{1 + idx}/{len(seeded_coordinators)}] Run dataset: {args.dataset}')
        args.coordinator = coordinator
        args.folder_results = init_folder_results(tmp_folder_results,
                                                  seed)

        run_graph_analysis(coordinator,
                           folder_results=args.folder_results,
                           is_test_graphs=args.test_graphs,
                           is_all_graphs=args.all_graphs)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--folder-data', type=str, required=True,
                        help='(e.g. ../../data/proteins/100/ )')
    parser.add_argument('--folder-results', type=str, required=True,
                        help='(e.g. --folder-results ./results/ )')
    parser.add_argument('--specific-seed', type=str, default=False)

    parser.add_argument('--test-graphs', action='store_true', default=False,
                        help='if present run for the test graphs')
    parser.add_argument('--all-graphs', action='store_true', default=False,
                        help='if present run for all the graphs')

    args = parser.parse_args()

    main(args)
