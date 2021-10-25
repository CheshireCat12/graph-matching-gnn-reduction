from argparse import ArgumentParser

from bunch import Bunch

from experiments_gnn_embedding.run_knn_gnn_embedding import run_knn_gnn_embedding
from graph_pkg.utils.functions.load_config import load_config

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


def run_experiment(args):
    parameters = load_config(args.exp)
    if args.all:
        for dataset in __DATASETS:
            args.dataset = dataset

            _run(args, parameters)
    else:
        _run(args, parameters)

    print_fancy_title('Final')


def _run(args, parameters):
    # Fusion the selected dataset parameters with the general parameters
    parameters = Bunch({**parameters[args.dataset], **parameters['general']})

    print_fancy_title('Parameters')
    print(parameters)

    print_fancy_title('Run')
    __EXPERIMENTS_GNN[args.exp](parameters)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')
    parser.add_argument('-e', '--exp', type=str, required=True,
                        choices=__EXPERIMENTS_GNN.keys(),
                        help='Choose the experiment to run.')
    parser.add_argument('-d', '--dataset', type=str,
                        default='letter',
                        choices=['letter', 'AIDS', 'mutagenicity', 'NCI1',
                                 'proteins_tu', 'enzymes',
                                 'collab', 'reddit_binary', 'IMDB_binary'],
                        help='Choose the dataset.')
    parser.add_argument('-a', '--all', type=bool,
                        default=False,
                        choices=[True, False],
                        help='Run on all available datasets.')
    args = parser.parse_args()
    run_experiment(args)
