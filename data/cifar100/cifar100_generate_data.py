"""
Download CIFAR-10 dataset, and splits it among clients
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import ConcatDataset
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

from cifar100_utils import split_dataset_by_labels, pathological_non_iid_split, pachinko_allocation_split


N_FINE_LABELS = 100
N_COARSE_LABELS = 20
N_COMPONENTS = 3
SEED = 22116
RAW_DATA_PATH = "/home/THY/cifar/cifar100/raw_data/"
PATH = "/home/THY/cifar/cifar100/all_data/"

COARSE_LABELS =\
    np.array([
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
        3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
        0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
        16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
        2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
        18, 1, 2, 15, 6, 0, 17, 8, 14, 13
    ])


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True)
    parser.add_argument(
        '--pathological_split',
        help='if selected, the dataset will be split as in'
             '"Communication-Efficient Learning of Deep Networks from Decentralized Data";'
             'i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes',
        action='store_true'
    )
    parser.add_argument(
        '--pachinko_allocation_split',
        help='if selected, the dataset will be split using Pachinko allocation,'
             'see "Adaptive Federated Optimization"__(https://arxiv.org/abs/2003.00295)',
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters;',
        type=int,
        default=N_COMPONENTS
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;'
             'used with `pachinko_allocation_split` and `_by_labels`;'
             'default is 0.4',
        type=float,
        default=0.4)
    parser.add_argument(
        '--beta',
        help='parameter controlling tasks dissimilarity, the smaller beta is the more tasks are dissimilar;'
             'used with `pachinko_allocation_split`; default is 10.0'
             'default is 10.0',
        type=float,
        default=10.
    )
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--val_frac',
        help='fraction in validation set (from train set); default: 0.0;',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=SEED)

    return parser.parse_args()


def main():
    args = parse_args()

    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset =\
        ConcatDataset([
            # CIFAR100(root=RAW_DATA_PATH, download=True, train=True, transform=transform),
            # CIFAR100(root=RAW_DATA_PATH, download=False, train=False, transform=transform)
            CIFAR100(root=RAW_DATA_PATH, download=False, train=True),
            CIFAR100(root=RAW_DATA_PATH, download=False, train=False)
        ])

    if args.pathological_split:
        clients_indices =\
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=N_FINE_LABELS,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed
            )
    elif args.pachinko_allocation_split:
        clients_indices = \
            pachinko_allocation_split(
                dataset=dataset,
                n_clients=args.n_tasks,
                coarse_labels=COARSE_LABELS,
                n_fine_labels=N_FINE_LABELS,
                n_coarse_labels=N_COARSE_LABELS,
                alpha=args.alpha,
                beta=args.beta,
                frac=args.s_frac,
                seed=args.seed
            )
    else:
        clients_indices = \
            split_dataset_by_labels(
                dataset=dataset,
                n_classes=N_FINE_LABELS,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed
            )

    if args.test_tasks_frac > 0:
        train_clients_indices, test_clients_indices = \
            train_test_split(
                clients_indices,
                test_size=args.test_tasks_frac,
                random_state=args.seed
            )
    else:
        train_clients_indices, test_clients_indices = clients_indices, []

    # os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    # os.makedirs(os.path.join(PATH, "test"), exist_ok=True)
    if args.pachinko_allocation_split:
        path0 = os.path.join(PATH, "alpha{}_beta{}".format(args.alpha, args.beta))
        if args.test_tasks_frac > 0:
            path0 = os.path.join(PATH, "alpha{}_beta{}_unseen{}".format(args.alpha, args.beta, args.test_tasks_frac))
    if args.pathological_split:
        path0 = os.path.join(PATH, "classesPerClient{}".format(args.n_shards))
        if args.test_tasks_frac > 0:
            path0 = os.path.join(PATH, "classesPerClient{}_unseen{}".format(args.n_shards, args.test_tasks_frac))

    os.makedirs(path0, exist_ok=True)

    cifar100_train = \
        CIFAR100(
            root=RAW_DATA_PATH,
            train=True, download=False
        )

    cifar100_test = \
        CIFAR100(
            root=RAW_DATA_PATH,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])
    train_clients = {}
    test_clients = {}
    col_name = ['fine'+str(i) for i in range(N_FINE_LABELS)] + ['coarse' + str(i) for i in range(N_COARSE_LABELS)]
    label_ratio = pd.DataFrame(columns=col_name)
    label_ratio_test = pd.DataFrame(columns=col_name)
    # for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
    for client_id, indices in enumerate(train_clients_indices):
        # client_path = os.path.join(PATH, mode, "task_{}".format(client_id))
        # os.makedirs(client_path, exist_ok=True)
        #
        train_indices, test_indices =\
            train_test_split(
                indices,
                train_size=args.tr_frac,
                random_state=args.seed
            )
        x_train = cifar100_data[train_indices]
        y_train = cifar100_targets[train_indices]
        # train_data = Data.TensorDataset(torch.from_numpy(x_train.numpy()), torch.tensor(y_train.numpy(), dtype=torch.long))
        train_data = Data.TensorDataset(x_train.permute(0,3,1,2).to(torch.float32), torch.tensor(y_train.numpy(), dtype=torch.long))
        train_clients[str(client_id)] = train_data

        x_test = cifar100_data[test_indices]
        y_test = cifar100_targets[test_indices]
        test_data = Data.TensorDataset(x_test.permute(0,3,1,2).to(torch.float32), torch.tensor(y_test.numpy(), dtype=torch.long))
        test_clients[str(client_id)] = test_data

        a1 = Counter(y_train.numpy())
        a2 = []
        a3 = {k: list() for k in range(N_COARSE_LABELS)}
        for k in range(N_FINE_LABELS):
            a2.append(a1[k])
            a3[COARSE_LABELS[k]].append(a1[k])
        a3 = [sum(v) for k,v in a3.items()]
        label_ratio.loc[client_id] = a2 + a3
        label_ratio.to_excel(path0 + '/label_ratio.xlsx', index=False)

        a1 = Counter(y_test.numpy())
        a2 = []
        a3 = {k: list() for k in range(N_COARSE_LABELS)}
        for k in range(N_FINE_LABELS):
            a2.append(a1[k])
            a3[COARSE_LABELS[k]].append(a1[k])
        a3 = [sum(v) for k, v in a3.items()]
        label_ratio_test.loc[client_id] = a2 + a3
        label_ratio_test.to_excel(path0 + '/label_ratio_test.xlsx', index=False)

    if test_clients_indices != []:
        unseen_clients = {}
        for client_id, indices in enumerate(test_clients_indices):
            x_new = cifar100_data[indices]
            y_new = cifar100_targets[indices]
            # train_data = Data.TensorDataset(torch.from_numpy(x_train.numpy()), torch.tensor(y_train.numpy(), dtype=torch.long))
            train_data = Data.TensorDataset(x_new.permute(0, 3, 1, 2).to(torch.float32),
                                            torch.tensor(y_new.numpy(), dtype=torch.long))
            unseen_clients[str(client_id)] = train_data

    if test_clients_indices == []:
        torch.save({'traindataset': train_clients, 'testdataset': test_clients}, path0 + '/torchdata')
    else:
        torch.save({'traindataset': train_clients, 'testdataset': test_clients, 'unseendataset': unseen_clients}, path0 + '/torchdata')



if __name__ == "__main__":
    main()

