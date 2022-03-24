import subprocess

from itertools import product

datasets = ['PROTEINS', 'DD', 'ENZYMES', 'NCI1', 'Mutagenicity']
level_reduction = ['100', '50', '25_freeze']
seed = '1860'

for dataset, lvl_red in product(datasets, level_reduction):
    command = f'python main.py ' \
              f'--dataset {dataset.lower()} ' \
              f'--folder-data ../../data_old/{dataset}/{lvl_red}/ ' \
              f'--folder-results ./results/{dataset.lower()}/analysis_refused/{lvl_red}_all/ ' \
              f'--specific-seed {seed} ' \
              f'--all-graphs'

    subprocess.call(command, shell=True)
