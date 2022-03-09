import os.path
from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from graph_analysis.constants.dataset_constants import DATASETS
from graph_analysis.run_analysis import run_graph_analysis

def main(args):
    coordinator = DATASETS[args.dataset]['coordinator']
    dataset_folders = os.path.join(coordinator['folder_dataset'], '*')
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
    Path(folder_results).mkdir(parents=True, exist_ok=True)


    # Merge the command line parameters with the constant parameters
    arguments = Namespace(**vars(args),
                          **{'coordinators': coordinators,
                             'current_coordinator': None,
                             'folder_results': folder_results})

    filename = os.path.join(folder_results, 'results_general.json')


    for idx, coordinator in enumerate(arguments.coordinators):
        current_exp = f'exp_{coordinator["folder_dataset"].split("/")[-3]}'
        print(f'[{1+idx}/{len(arguments.coordinators)}] Run dataset: {current_exp}')
        arguments.current_coordinator = coordinator

        run_graph_analysis(arguments)



if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS)
    parser.add_argument('--data-folder', type=str)
    parser.add_argument('--specific-seed', type=str, default=False)

    parser.add_argument('--name-experiment', type=str, required=True,
                        help='Specify the experiment name under which to save the experiment')
    args = parser.parse_args()

    main(args)
