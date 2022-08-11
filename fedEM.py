
from utils.utils import *
from utils.constants import *
from utils.args import *
from utils.optim import get_optimizer, get_lr_scheduler
from models import CNNMnist, get_vgg16bn, get_resnet18
import copy
import pandas as pd
import numpy as np
import os
import torch
import torch.utils.data as Data
from tqdm import tqdm
from img_utils_linux import Data_Process, Shakespeare_util


def init_clients(args_):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_root: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators = [], [], []
    train_iterators_t, val_iterators_t, test_iterators_t = [], [], []
    client_list = None
    if args_.dataset == 'femnist':
        TARGET_PATH = args_.femnist_torchpath
        file_names_list = os.listdir(TARGET_PATH)
        if args_.new_test:
            client_new_file = np.random.choice(file_names_list, size = round(len(file_names_list) * args_.test_tasks_frac), replace=False)
            file_names_list = list(set(file_names_list) - set(client_new_file))
        for idx, file_name in enumerate(tqdm(file_names_list)):
            train = torch.load(os.path.join(TARGET_PATH, file_name, 'train.pt'))
            n = len(train[1])
            train = Data.TensorDataset(train[0].unsqueeze(1).to(torch.float32), train[1])
            train_loader = torch.utils.data.DataLoader(train, batch_size=args_.bz, shuffle=True)
            train_iterators.append(train_loader)
            val_iterators.append(torch.utils.data.DataLoader(train, batch_size=n))

            test = torch.load(os.path.join(TARGET_PATH, file_name, 'test.pt'))
            n = len(test[1])
            test = Data.TensorDataset(test[0].unsqueeze(1).to(torch.float32), test[1])
            test_iterators.append(torch.utils.data.DataLoader(test, batch_size=n))

        if args_.new_test:
            for idx, file_name in enumerate(tqdm(client_new_file)):
                train = torch.load(os.path.join(TARGET_PATH, file_name, 'train.pt'))
                n = len(train[1])
                train = Data.TensorDataset(train[0].unsqueeze(1).to(torch.float32), train[1])
                train_loader = torch.utils.data.DataLoader(train, batch_size=args_.bz, shuffle=True)
                train_iterators_t.append(train_loader)
                val_iterators_t.append(torch.utils.data.DataLoader(train, batch_size=n))

                test = torch.load(os.path.join(TARGET_PATH, file_name, 'test.pt'))
                n = len(test[1])
                test = Data.TensorDataset(test[0].unsqueeze(1).to(torch.float32), test[1])
                test_iterators_t.append(torch.utils.data.DataLoader(test, batch_size=n))
        client_list = [i for i in range(len(train_iterators))]
        newclient_list = [i for i in range(len(train_iterators_t))]

    elif args_.dataset == 'shakespeare':
        shakespeare_util = Shakespeare_util(args_)
        train_clients, test_clients = shakespeare_util.get_dataloader()
        model = shakespeare_util.get_model()
        client_list = train_clients.keys()
        if args.new_test:
            new_clients_train, new_clients_test = {}, {}
            newclient_list = np.random.choice(client_list, size = round(len(client_list) * args_.test_tasks_frac), replace=False)
            for client_i in client_list:
                if client_i in newclient_list:
                    new_clients_train[client_i] = train_clients[client_i]
                    new_clients_test[client_i] = test_clients[client_i]
                    del train_clients[client_i]
                    del test_clients[client_i]
            client_list = list(set(client_list) - set(newclient_list))
        train_iterators = [torch.utils.data.DataLoader(v, batch_size=args_.bz, shuffle=True) for k,v in train_clients.items()]
        for k in client_list:
            val_iterators.append(torch.utils.data.DataLoader(train_clients[k], batch_size=len(train_clients[k])))
            test_iterators.append(torch.utils.data.DataLoader(test_clients[k], batch_size=len(test_clients[k])))
        train_iterators_t = [torch.utils.data.DataLoader(v, batch_size=args_.bz, shuffle=True) for k,v in new_clients_train.items()]
        for k in newclient_list:
            val_iterators_t.append(torch.utils.data.DataLoader(new_clients_train[k], batch_size=len(new_clients_train[k])))
            test_iterators_t.append(torch.utils.data.DataLoader(new_clients_test[k], batch_size=len(new_clients_test[k])))

    elif args_.dataset == 'cifar':
        data_Process = Data_Process(args_)
        train_clients, test_clients = data_Process.fetch_data()
        client_list = list(train_clients.keys())
        if args_.new_test:
            new_clients_train, new_clients_test = {}, {}
            newclient_list = np.random.choice(client_list, size = round(len(client_list) * args_.test_tasks_frac), replace=False)
            for client_i in client_list:
                if client_i in newclient_list:
                    new_clients_train[client_i] = train_clients[client_i]
                    new_clients_test[client_i] = test_clients[client_i]
                    del train_clients[client_i]
                    del test_clients[client_i]
            client_list = list(set(client_list) - set(newclient_list))
        train_iterators = [torch.utils.data.DataLoader(v, batch_size=args_.bz, shuffle=True) for k, v in
                           train_clients.items()]
        for k in client_list:
            val_iterators.append(torch.utils.data.DataLoader(train_clients[k], batch_size = len(train_clients[k])))
            test_iterators.append(torch.utils.data.DataLoader(test_clients[k], batch_size = len(test_clients[k])))
        train_iterators_t = [torch.utils.data.DataLoader(v, batch_size=args_.bz, shuffle=True) for k, v in
                             new_clients_train.items()]
        for k in newclient_list:
            val_iterators_t.append(torch.utils.data.DataLoader(new_clients_train[k], batch_size=len(new_clients_train[k])))
            test_iterators_t.append(torch.utils.data.DataLoader(new_clients_test[k], batch_size=len(new_clients_test[k])))

    elif args_.dataset == 'cifar100':
        cifar100_path = os.path.join(args_.cifar100_path, "classesPerClient{}".format(args_.n_shards)) + '/torchdata'
        data_load = torch.load(cifar100_path)
        train_clients = data_load['traindataset']
        test_clients = data_load['testdataset']
        client_list = list(train_clients.keys())
        if args_.new_test:
            new_clients_train, new_clients_test = {}, {}
            newclient_list = np.random.choice(client_list, size = round(len(client_list) * args_.test_tasks_frac), replace=False)
            for client_i in client_list:
                if client_i in newclient_list:
                    new_clients_train[client_i] = train_clients[client_i]
                    new_clients_test[client_i] = test_clients[client_i]
                    del train_clients[client_i]
                    del test_clients[client_i]
            client_list = list(set(client_list) - set(newclient_list))

        train_iterators = [torch.utils.data.DataLoader(v, batch_size=args_.bz, shuffle=True) for k, v in
                           train_clients.items()]
        for k in client_list:
            val_iterators.append(torch.utils.data.DataLoader(train_clients[k], batch_size=len(train_clients[k])))
            test_iterators.append(torch.utils.data.DataLoader(test_clients[k], batch_size=len(test_clients[k])))
        train_iterators_t = [torch.utils.data.DataLoader(v, batch_size=args_.bz, shuffle=True) for k, v in
                             new_clients_train.items()]
        for k in newclient_list:
            val_iterators_t.append(torch.utils.data.DataLoader(new_clients_train[k], batch_size=len(new_clients_train[k])))
            test_iterators_t.append(torch.utils.data.DataLoader(new_clients_test[k], batch_size=len(new_clients_test[k])))

    print("===> Initializing learner models..")
    learners_model, learners_optim, learners_optim_lrsched = [], [], []
    device = torch.device('cuda' if (torch.cuda.is_available()) & (args_.device == 'cuda') else 'cpu')

    def get_model():
        if args_.dataset == "cifar":
            model = get_resnet18(n_classes=10)
            return model
        elif args_.dataset == "cifar100":
            model = get_vgg16bn(n_classes = 100)
            return model
        elif args_.dataset == "emnist" or args_.dataset == "femnist":
            model = CNNMnist(num_classes=62)
            return model

    if args_.dataset != 'shakespeare':
        model = get_model()
    def get_model_dim():
        param_list = []
        for param in model.parameters():
            param_list.append(param.data.view(-1, ))
        return int(torch.cat(param_list).shape[0])
    model_dim = get_model_dim()

    for i in range(args_.n_learners):
        learners_model.append(copy.deepcopy(model).to(device))
        optimizer = \
            get_optimizer(
                optimizer_name=args_.optimizer,
                model=learners_model[i],
                lr_initial=args_.lr,
                mu=args_.mu
            )
        learners_optim.append(optimizer)
        lr_scheduler = \
            get_lr_scheduler(
                optimizer=optimizer,
                scheduler_name=args_.lr_scheduler,
                n_rounds=args_.n_rounds
            )
        learners_optim_lrsched.append(lr_scheduler)


    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator, client_name) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators, client_list), total=len(train_iterators))):
        model_dict = model.state_dict()
        clients_model = [model_dict] * args_.n_learners

        if train_iterator is None or test_iterator is None:
            continue

        learners_ensemble =\
            get_learners_ensemble(
                n_learners=args_.n_learners,
                clients_model = clients_model,
                model_dim = model_dim,
                name=args_.experiment,
                device=args_.device,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                seed=args_.seed,
            )

        client = get_client(
            client_name = client_name,
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            # clientsModel = clients_model[task_id],
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger='',
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients
        )
        clients_.append(client)

    test_clients_ = []
    for task_id, (train_iterator_t, val_iterator_t, test_iterator_t, client_name) in \
            enumerate(tqdm(zip(train_iterators_t, val_iterators_t, test_iterators_t, newclient_list), total=len(train_iterators_t))):
        model_dict = model.state_dict()
        clients_model = [model_dict] * args_.n_learners

        if train_iterator_t is None or test_iterator_t is None:
            continue

        learners_ensemble =\
            get_learners_ensemble(
                n_learners=args_.n_learners,
                clients_model = clients_model,
                model_dim = model_dim,
                name=args_.experiment,
                device=args_.device,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                seed=args_.seed,
            )

        client_t = get_client(
            client_name = client_name,
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            # clientsModel = clients_model[task_id],
            q=args_.q,
            train_iterator=train_iterator_t,
            val_iterator=val_iterator_t,
            test_iterator=test_iterator_t,
            logger='',
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients
        )
        test_clients_.append(client_t)

    return learners_model, clients_model, model_dim, learners_optim, learners_optim_lrsched, clients_, test_clients_, client_list


def run_experiment(args_):
    torch.manual_seed(args_.seed)

    print("==> Clients initialization..")
    learners_model, clients_model, model_dim, learners_optim, learners_optim_lrsched, clients, test_clients,\
        client_list = init_clients(args_)

    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            clients_model = [copy.deepcopy(item) for item in clients_model],
            model_dim=model_dim,
            name=args_.experiment,
            device=args_.device,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            seed=args_.seed,
        )

    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    #--------------------------------------------------------
    if (args_.dataset == 'femnist') or (args_.dataset == 'shakespeare'):
        file_path = os.path.join(args_.result_path, args_.method, args_.dataset)
    elif args_.dataset == 'cifar100':
        file_path = os.path.join(args_.result_path, args_.method, args_.dataset, 'd_{}'.format(args_.n_shards))
    else:
        file_path = os.path.join(args_.result_path, args_.method, args_.dataset, 'dirichlet_{}'.format(args_.dirichlet_parameter))
    if args_.new_test:
        file_path = os.path.join(file_path, "new_test")
    os.makedirs(file_path, exist_ok=True)
    # 将args写入txt
    f = open(file_path + '/args.txt', 'a')
    for k in args_.__dict__:
        f.write('\n' + k + ": " + str(args_.__dict__[k]))
    f.close()

    if client_list is None:
        client_list = [i for i in range(len(clients))]
    loss_train, loss_test = pd.DataFrame(columns=client_list), pd.DataFrame(columns=client_list)
    acc_train, acc_test = pd.DataFrame(columns=client_list), pd.DataFrame(columns=client_list)
    fed_metrics = pd.DataFrame(columns=['loss_train', 'loss_test', 'acc_train', 'acc_test'])
    loss_train_new, loss_test_new = pd.DataFrame(columns=client_list), pd.DataFrame(columns=client_list)
    acc_train_new, acc_test_new = pd.DataFrame(columns=client_list), pd.DataFrame(columns=client_list)
    fed_metrics_new = pd.DataFrame(columns=['loss_train', 'loss_test', 'acc_train', 'acc_test'])

    aggregator =\
        get_aggregator(
            args=args_,
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger='',
            global_test_logger='',
            test_clients = test_clients,
            verbose=args_.verbose,
            seed=args_.seed,
            file_path = file_path,
            loss_train_df = loss_train,
            loss_test_df = loss_test,
            acc_train_df = acc_train,
            acc_test_df = acc_test,
            fed_metrics_df = fed_metrics,
            loss_train_new_df=loss_train_new,
            loss_test_new_df=loss_test_new,
            acc_train_new_df=acc_train_new,
            acc_test_new_df=acc_test_new,
            fed_metrics_new_df = fed_metrics_new
        )

    print("Training..")

    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    while current_round <= args_.n_rounds:
        print('='*30 + 'round:'+str(current_round) + '='*30)
        aggregator.mix(learners_model, learners_optim, learners_optim_lrsched)

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round

    if "save_path" in args_:
        save_root = os.path.join(args_.save_path)

        os.makedirs(save_root, exist_ok=True)
        aggregator.save_state(save_root)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    run_experiment(args)