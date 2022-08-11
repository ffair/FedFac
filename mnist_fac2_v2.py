# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:58:22 2021

@author: THY
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import copy
import torch
import torch.utils.data as Data
from my_models import get_resnet18, get_vgg16bn
from img_utils_linux import CNNMnist, LocalUpdate, test_img, Partial_fed_avg, personalize_FedAvg, \
    Data_Process, Shakespeare_util, General_test, General_test_ensemble
import os


import argparse
parser = argparse.ArgumentParser()
# path
parser.add_argument("--femnist_torchpath", type= str, default= "/home/THY/mnist/femnist/all_data/train", help="")
parser.add_argument("--result_path", type= str, default= '/home/THY/fl_fac', help="")
parser.add_argument("--mnist_cached_file_pfl", type= str, default= '/home/THY/mnist/torchdata', help="")
parser.add_argument("--mnist_rawdata", type= str, default= '/home/THY/mnist', help="")
parser.add_argument("--cifar_cached_file_pfl", type= str, default= '/home/THY/cifar/torchdata', help="")
parser.add_argument("--cifar_rawdata", type= str, default= '/home/THY/cifar/CIFAR/cifar-10-python', help="")
parser.add_argument("--shakespeare_root_path", type= str, default= '/home/THY/shakespeare/all_data/train', help="")
parser.add_argument("--digits_path", type= str, default= '/home/THY/digit_dataset', help="")
parser.add_argument("--cifar100_path", type= str, default='/home/THY/cifar/cifar100/all_data', help="")

# fac related
parser.add_argument("--direct_eigDecom", action = 'store_true', help="if true, execute eigen decomposition directly")
parser.add_argument("--given_threshold", action = 'store_true', help="if true, use given_threshold as threshold, else mean")
parser.add_argument("--given_threshold_v", type= float, default= np.Inf, help="given threshold value for factor analysis")
parser.add_argument("--threshold_p", action = 'store_true', help="if true, use percentile as threshold, else mean")
parser.add_argument("--threshold", type= int, default= 50, help="threshold for deciding whether a channel is private or shared")
parser.add_argument("--partial_fed", action = 'store_true', help="whether take partial cnn")
parser.add_argument("--partial_cnn_layer", type= str, default = '1', help="'1':execute fac on the first layer;"
                                                                          "'2':execute fac on the second layer;"
                                                                          "'1+2':execute fac on the first and second layer")
parser.add_argument("--private_remove", action = 'store_true', help="masking private elements")
parser.add_argument("--shared_remove", action = 'store_true', help="masking shared elements")
parser.add_argument("--last_layer_not_share", action = 'store_true', help="keep the last layer from averaging")
parser.add_argument("--cov_thrhd", type= float, default= 0.85, help="the parameter kappa")

# assign clients
parser.add_argument("--dataset", type=str, default = 'cifar', choices = ['mnist', 'cifar', 'femnist',
                                                                         'shakespeare', 'cifar100'])
parser.add_argument("--devide_by_label", action = 'store_true', help="if true, clients are devided by labels of 0-9")
parser.add_argument("--sampling", type= str, default = 'dirichlet', choices= ['dirichlet', 'pathological_split'], help="sample scheme")
parser.add_argument("--dirichlet_parameter", type=float, default = 0.01, help="dirichlet parameter for sampling")
# cifar100
parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--sampling` is dirichlet;'
             'default is 2',
        type=int,
        default=2
    )

# FL setting
parser.add_argument("--seed", type = int, default = 1012, help="random seed")
parser.add_argument("--device", type=str, default='gpu', help="")
parser.add_argument("--batch_size", type=int, default = 32, help="local batch size")
parser.add_argument("--local_ep", type=int, default = 5, help="local epoches")
parser.add_argument("--rounds", type=int, default = 250, help='total communication rounds')
parser.add_argument("--num_clients", type=int, default = 100, help="number of total clients")
parser.add_argument("--num_clients_frac", type= float, default = 0.1, help="number of clients selected for each round")

# net
parser.add_argument("--nettype", type = str, default = 'cnnmnist', help="cifar: 'pre_resnet18'; femnist: 'cnnmnist'")
# femnist cnn
parser.add_argument("--num_channels", type= str, default='32,64', help="number of cnn channels of each cnn layer of mnist model")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes")

# optimizer
parser.add_argument("--optimizer", type = str, default = 'adam', help="local optimizer: adam, sgd ")
parser.add_argument("--weight_decay", type = float, default = 0, help="weight_decay of adam, 0.0005, 0.00001")
parser.add_argument("--lr", type = float, default= 0.001, help="learning rate")
parser.add_argument("--lr_scheduler", action = 'store_true', help="learning rate scheduler")
parser.add_argument("--lr_drop", type = int, default = 20, help="frequency for lr to drop")
parser.add_argument("--lr_gamma", type = float, default = 0.9, help=" ")
parser.add_argument("--lmbda", type = float, default = 0.9, help="lr = args.lmbda ** (rnd // self.args.lr_drop")

# method
parser.add_argument("--method", type = str, default = 'fed_fac', choices=['fed_fac', 'fed', 'fedper', 'fedprox', 'lg'])
# fedprox
parser.add_argument("--fedprox", action = 'store_true', help = "use fedprox algorithm")
parser.add_argument("--mu", type = float, default = 0.1, help="mu/2 * ||w - w^t||^2")

# model saving and loading
parser.add_argument("--save_model", action = 'store_true', help = "whether save model or not")
parser.add_argument("--load_model", action = 'store_true', help = "whether load model or not")
parser.add_argument("--early_stop", action = 'store_true', help = "whether early stopping")

## generalization test
parser.add_argument("--new_test", action = 'store_true', help = "whether evaluate on unseen clients")
parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
# fac generalization
parser.add_argument("--fac_newtest_priv", type = str, default = 'Avg', choices = ["train-fix", "Avg", "fix-tune"],
                    help="the method for fac generalization, it takes Avg or local_train")
parser.add_argument("--general_ensemble", action = 'store_true', help = "use ensemble for generalization test, for method == lg/fac")
args = parser.parse_args()

for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    device = torch.device('cuda' if (torch.cuda.is_available()) & (args.device == 'gpu') else 'cpu')
    setup_seed(args.seed)
    # load data
    if args.dataset == 'femnist':
        TARGET_PATH = args.femnist_torchpath
        file_names_list = os.listdir(TARGET_PATH)
        train_clients, test_clients = {}, {}
        if args.new_test:
            client_new_file = np.random.choice(file_names_list, size = round(len(file_names_list) * args.test_tasks_frac), replace=False)
            file_names_list = list(set(file_names_list) - set(client_new_file))
        for idx, file_name in enumerate(tqdm(file_names_list)):
            train = torch.load(os.path.join(TARGET_PATH, file_name, 'train.pt'))
            test = torch.load(os.path.join(TARGET_PATH, file_name, 'test.pt'))
            train_clients[str(idx)] = Data.TensorDataset(train[0].unsqueeze(1).to(torch.float32), train[1])
            test_clients[str(idx)] = Data.TensorDataset(test[0].unsqueeze(1).to(torch.float32), test[1])
        client_list = list(train_clients.keys())
        client_model = {}
        model = CNNMnist(args)
        for client_i in client_list:
            client_model[client_i] = model.state_dict()
        if args.new_test:
            new_clients_train, new_clients_test = {},{}
            for idx, file_name in enumerate(tqdm(client_new_file)):
                train = torch.load(os.path.join(TARGET_PATH, file_name, 'train.pt'))
                test = torch.load(os.path.join(TARGET_PATH, file_name, 'test.pt'))
                new_clients_train[str(idx)] = Data.TensorDataset(train[0].unsqueeze(1).to(torch.float32), train[1])
                new_clients_test[str(idx)] = Data.TensorDataset(test[0].unsqueeze(1).to(torch.float32), test[1])
            newclient_list = list(new_clients_train.keys())
            newclient_model = {}
            for client_i in newclient_list:
                newclient_model[client_i] = model.state_dict()
    elif args.dataset == 'shakespeare':
        shakespeare_util = Shakespeare_util(args)
        train_clients, test_clients = shakespeare_util.get_dataloader()
        client_list = list(train_clients.keys())
        if args.new_test:
            new_clients_train, new_clients_test = {},{}
            newclient_list = np.random.choice(client_list, size = round(len(client_list) * args.test_tasks_frac), replace=False)
            for client_i in client_list:
                if client_i in newclient_list:
                    new_clients_train[client_i] = train_clients[client_i]
                    new_clients_test[client_i] = test_clients[client_i]
                    del train_clients[client_i]
                    del test_clients[client_i]
            client_list = list(set(client_list) - set(newclient_list))
        model = shakespeare_util.get_model()
        client_model = {}
        for client_i in client_list:
            client_model[client_i] = model.state_dict()
        if args.new_test:
            newclient_model = {}
            for client_i in newclient_list:
                newclient_model[client_i] = model.state_dict()
    elif args.dataset == 'cifar100':
        if args.sampling == 'dirichlet':
            cifar100_path = os.path.join(args.cifar100_path, "alpha{}_beta{}".format(args.alpha, args.beta)) + '/torchdata'
        else:
            cifar100_path = os.path.join(args.cifar100_path, "classesPerClient{}".format(args.n_shards)) + '/torchdata'
        data_load = torch.load(cifar100_path)
        train_clients = data_load['traindataset']
        test_clients = data_load['testdataset']
        client_list = list(train_clients.keys())
        model = get_vgg16bn(args.num_classes)
        client_model = {}
        if args.new_test:
            new_clients_train, new_clients_test = {}, {}
            newclient_list = np.random.choice(client_list, size = round(len(client_list) * args.test_tasks_frac), replace=False)
            for client_i in client_list:
                if client_i in newclient_list:
                    new_clients_train[client_i] = train_clients[client_i]
                    new_clients_test[client_i] = test_clients[client_i]
                    del train_clients[client_i]
                    del test_clients[client_i]
            client_list = list(set(client_list) - set(newclient_list))
        for client_i in client_list:
            client_model[client_i] = model.state_dict()
        if args.new_test:
            newclient_model = {}
            for client_i in newclient_list:
                newclient_model[client_i] = model.state_dict()
    else:
        data_Process = Data_Process(args)
        train_clients, test_clients = data_Process.fetch_data()
        client_list = list(train_clients.keys())
        if args.new_test:
            new_clients_train, new_clients_test = {}, {}
            newclient_list = np.random.choice(client_list, size = round(len(client_list) * args.test_tasks_frac), replace=False)
            for client_i in client_list:
                if client_i in newclient_list:
                    new_clients_train[client_i] = train_clients[client_i]
                    new_clients_test[client_i] = test_clients[client_i]
                    del train_clients[client_i]
                    del test_clients[client_i]
            client_list = list(set(client_list) - set(newclient_list))
        if args.dataset == 'mnist':
            model = CNNMnist(args)
        if args.dataset == 'cifar':
            model = get_resnet18(args.num_classes)

        client_model = {}
        for client_i in client_list:
            client_model[client_i] = model.state_dict()
        if args.new_test:
            newclient_model = {}
            for client_i in newclient_list:
                newclient_model[client_i] = model.state_dict()

    # FL initial state
    layer_name, layer_size, global_param_d = [], [], {}

    for n, p in model.named_parameters():
        layer_name.append(n)
        layer_size.append(len(p.size()))
        global_param_d[n] = p

    # FedProx
    global_param = []
    for k, v in global_param_d.items():
        global_param.append(torch.flatten(v, 0))
    if args.device == 'gpu':
        global_param = [item.to(device) for item in global_param]

    # get share and partial share keys
    l_n, fac_type = {}, {}
    if (args.nettype == 'cnnmnist') or (args.dataset == 'sent140'):
        for i in range(len(layer_name) // 2):
            l_n[str(i + 1)] = layer_name[2 * i:2 * (i + 1)]
        for k, v in l_n.items():
            if 'conv' in v[0]:
                fac_type[v[0]] = 'conv'
            else:
                fac_type[v[0]] = 'dense'

    if args.dataset == 'digits':
        for i in range(len(layer_name) // 4):
            l_n[str(i + 1)] = layer_name[4 * i:4 * (i + 1)]
        l_n[str(6)] = layer_name[-2:]
        for k, v in l_n.items():
            if 'conv' in v[0]:
                fac_type[v[0]] = 'conv'
            else:
                fac_type[v[0]] = 'dense'

    if args.dataset == 'shakespeare':
        l_n['1'] = layer_name[0:1]
        l_n['2'] = layer_name[1:5]
        l_n['3'] = layer_name[5:9]
        l_n['4'] = layer_name[9:11]
        l_n['5'] = layer_name[11:13]
        fac_type[l_n['4'][0]] = 'dense'
        fac_type[l_n['5'][0]] = 'dense'

    if args.dataset == 'cifar':
        for i in range(len(layer_name) // 3):
            l_n[str(i + 1)] = layer_name[3 * i:3 * (i + 1)]
        l_n[str(21)] = layer_name[-2:]
        for k, v in l_n.items():
            if ('conv' in v[0]) or ('downsample' in v[0]):
                fac_type[v[0]] = 'conv'
            else:
                fac_type[v[0]] = 'dense'
    if args.dataset == 'cifar100':
        for i in range((len(layer_name)-6) // 4):
            l_n[str(i + 1)] = layer_name[4 * i:4 * (i + 1)]
        for i in range(3):
            l_n[str(i + 14)] = layer_name[2 * i+52:2 * (i + 1)+52]
        for k, v in l_n.items():
            if 'features' in v[0]:
                fac_type[v[0]] = 'conv'
            else:
                fac_type[v[0]] = 'dense'

    pcl = args.partial_cnn_layer.split('+')
    partial_shared_layer_key = []
    nsk = []
    for i in pcl:
        partial_shared_layer_key.append(l_n[i][0])
        nsk = nsk + l_n[i]
    if args.last_layer_not_share:
        nsk = nsk + l_n[list(l_n.keys())[-1]]

    if args.partial_fed or args.method == 'lg' or args.method == 'fedrep':
        shared_layer_keys = [item for item in layer_name if item not in nsk]
    elif args.method == 'fedbn':
        shared_layer_keys = [item for item in layer_name if 'bn' not in item]
    else:
        if args.last_layer_not_share:
            shared_layer_keys = [item for item in layer_name if item not in l_n[list(l_n.keys())[-1]]]
        else:
            shared_layer_keys = layer_name

    print('-'*20 + 'layer_name' + '-'*20)
    print(layer_name)
    print('-'*20 + 'partial_shared_layer_key' + '-'*20)
    print(partial_shared_layer_key)
    print('-'*20 + 'shared_layer_keys' + '-'*20)
    print(shared_layer_keys)

    ## set result path
    if args.devide_by_label:
        sampling = 'devide_by_label'
    else:
        sampling = args.sampling

    if args.threshold_p:
        threshold = args.threshold
    else:
        threshold = 'mean'

    if (args.dataset == 'femnist') or (args.dataset == 'shakespeare'):
        file_path = os.path.join(args.result_path, args.method, args.dataset)
    else:
        file_path = os.path.join(args.result_path, args.method, args.dataset, sampling)

    if args.sampling == 'dirichlet':
        sample_p = args.dirichlet_parameter
    else:
        sample_p = 'shards{}'.format(args.n_shards)

    if (args.dataset == 'femnist') or (args.dataset == 'shakespeare'):
        sample_p = 'none'

    if args.partial_fed:
        result_path = os.path.join(file_path, 'threshold{}_p{}_l{}_covthrhd{}'.format(threshold, sample_p, args.partial_cnn_layer, args.cov_thrhd))
    else:
        if args.method == 'fedprox':
            result_path = os.path.join(file_path, 'mu{}_p{}_r{}'.format(args.mu, sample_p, args.rounds))
        else:
            result_path = os.path.join(file_path, 'p{}_r{}'.format(sample_p, args.rounds))

    if args.new_test:
        result_path = os.path.join(result_path, "new_test")
    if args.shared_remove:
        result_path = os.path.join(result_path, "sr")
    if args.private_remove:
        result_path = os.path.join(result_path, "pr")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    f = open(result_path + '/args.txt', 'a')
    for k in args.__dict__:
        f.write('\n' + k + ": " + str(args.__dict__[k]))
    f.close()

    ##
    loss_train, loss_test = pd.DataFrame(columns = client_list),pd.DataFrame(columns = client_list)
    acc_train, acc_test = pd.DataFrame(columns = client_list), pd.DataFrame(columns = client_list)
    fed_metrics = pd.DataFrame(columns = ['loss_train', 'loss_test', 'acc_train', 'acc_test'])
    fed_metrics_all = pd.DataFrame(columns = ['loss_test', 'acc_test'])
    if args.new_test:
        loss_newtest = pd.DataFrame(columns = newclient_list)
        if args.general_ensemble:
            fed_metrics_new = pd.DataFrame(columns=['loss_test', 'acc_test_avg', 'acc_test_major'])
            acc_avg_newtest, acc_maj_newtest = pd.DataFrame(columns = newclient_list), pd.DataFrame(columns = newclient_list)
        else:
            acc_newtest = pd.DataFrame(columns = newclient_list)
            fed_metrics_new = pd.DataFrame(columns=['loss_test', 'acc_test'])

    partial_conv_shared, shared_layers_param = {},{}
    client_optimizer = {}

    def save_metrics():
        loss_train.to_excel(result_path + '/losstrain_{}_{}.xlsx'.format(args.nettype, args.optimizer))
        loss_test.to_excel(result_path + '/losstest_{}_{}.xlsx'.format(args.nettype, args.optimizer))
        acc_test.to_excel(result_path + '/acctest_{}_{}.xlsx'.format(args.nettype, args.optimizer))
        acc_train.to_excel(result_path + '/acctrain_{}_{}.xlsx'.format(args.nettype, args.optimizer))
        fed_metrics.to_excel(result_path + '/fedmetrics_{}_{}.xlsx'.format(args.nettype, args.optimizer))
        fed_metrics_all.to_excel(result_path + '/fedmetrics_all_{}_{}.xlsx'.format(args.nettype, args.optimizer))

    # save model
    def save_checkpoint():
        torch.save({'partial_conv_shared': partial_conv_shared}, os.path.join(result_path, "p_param" + ".pth"))
        torch.save(shared_layers_param, os.path.join(result_path, "s_param" + ".pth"))
        client_model_param = [client_model[i] for i in client_list]
        client_model_param = dict(zip(client_list, client_model_param))
        torch.save(client_model_param, os.path.join(result_path, 'client_model' + '.pth'))

    # early stop
    best_score = None
    delta = 0
    counter, patience = 0, 50
    early_stop = False

    num_clients_per_round = int(args.num_clients_frac * len(client_list))

    testdata_size = [len(test_clients[i]) for i in client_list]
    testdata_size = dict(zip(client_list, testdata_size))
    print('-'*10+'client_size'+'-'*10)
    print(len(client_list))
    ## training
    model.to(device)
    for rnd in range(args.rounds):
        print('-'*10+'Round {}'.format(rnd)+'-'*10)
        client_sample = np.random.choice(client_list, size = num_clients_per_round, replace=False)
        param_locals = {}
        train_data_size = [len(train_clients[i]) for i in client_sample]
        train_data_size = dict(zip(client_sample, train_data_size))
        test_data_size = [len(test_clients[i]) for i in client_sample]
        test_data_size = dict(zip(client_sample, test_data_size))
        loss_train_fed, loss_test_fed, acc_train_fed, acc_test_fed = 0,0,0,0

        for client in client_sample:
            model.load_state_dict(client_model[client])
            if rnd >= 1:
                if args.partial_fed:
                    # download global partial parameter
                    for key in partial_shared_layer_key:
                        partial_fed_avg = Partial_fed_avg(args, param_locals, train_data_size, key)
                        partial_fed_avg.partial_param_update(model, client_model[client], partial_conv_shared[key])
                        ## generalized testing
                        if args.shared_remove:
                            partial_fed_avg.partial_shared_removed(model, client_model[client], partial_conv_shared[key])
                        if args.private_remove:
                            partial_fed_avg.private_removed(model, client_model[client], partial_conv_shared[key])
                    # download global full averaged para
                    model_dict = model.state_dict()
                    model_dict.update(copy.deepcopy(shared_layers_param))
                    model.load_state_dict(model_dict)
                else:
                    # download global full averaged para
                    model_dict = client_model[client]
                    model_dict.update(copy.deepcopy(shared_layers_param))
                    model.load_state_dict(model_dict)

            localupdate = LocalUpdate(args = args, train_data = train_clients[client], net = model)
            if args.save_model:
                w, loss, acc, optim = localupdate.train(rnd, global_param, shared_layer_keys)
                client_optimizer[client] = copy.deepcopy(optim)
            else:
                w, loss, acc = localupdate.train(rnd, global_param, shared_layer_keys, shared_layers_param)
            client_model[client] = copy.deepcopy(w)
            param_locals[client] = copy.deepcopy(w)
            loss_train.loc[rnd, client] = loss
            acc_train.loc[rnd, client] = acc
            acc_train_fed += acc * train_data_size[client]
            loss_train_fed += loss * train_data_size[client]

        # partial aggregate
        if args.partial_fed:
            partial_conv_shared = {}
            newtest_priv_para = {} # new test private kernel parameter
            p_s_indx = {}
            for key in partial_shared_layer_key:
                partial_fed_avg = Partial_fed_avg(args, param_locals, train_data_size, key)
                priv_indx, share_indx = partial_fed_avg.get_s_p_indx(fac_type[key])
                p_s_indx[key] = [priv_indx, share_indx]
                partial_conv_shared[key] = partial_fed_avg.partial_para_avg(share_indx)
                # new test private kernel parameter
                newtest_priv_para[key] = partial_fed_avg.partial_para_avg(priv_indx)
                f = open(result_path + '/kernel_sindx_new.txt', 'a')
                f.write('\n' + str(rnd) + key + ": " + str(share_indx))
                f.close()

        # full 平均
        shared_layers_param = personalize_FedAvg(client_sample, param_locals, train_data_size, shared_layer_keys)

        # fedprox
        global_param = []
        for k, v in shared_layers_param.items():
            global_param.append(torch.flatten(v, 0))

        ## evaluate
        losstest_fed_all, acctest_fed_all = 0, 0
        for client in client_list:
            if args.partial_fed:
                # download global partial para
                for key in partial_shared_layer_key:
                    partial_fed_avg = Partial_fed_avg(args, param_locals, train_data_size, key)
                    partial_fed_avg.partial_param_update(model, client_model[client], partial_conv_shared[key])
                    if args.shared_remove:
                        partial_fed_avg.partial_shared_removed(model, client_model[client], partial_conv_shared[key])
                    if args.private_remove:
                        partial_fed_avg.private_removed(model, client_model[client], partial_conv_shared[key])
                # download global full avg para
                model_dict = model.state_dict()
                model_dict.update(copy.deepcopy(shared_layers_param))
                model.load_state_dict(model_dict)
            else:
                # download global full avg para
                model_dict = client_model[client]
                model_dict.update(copy.deepcopy(shared_layers_param))
                model.load_state_dict(model_dict)

            accuracy, loss = test_img(rnd, model, test_clients[client], args, shared_layers_param, False, global_param)
            loss_test.loc[rnd, client] = loss
            acc_test.loc[rnd, client] = accuracy
            losstest_fed_all += loss * testdata_size[client]
            acctest_fed_all += accuracy * testdata_size[client]
            if client in client_sample:
                loss_test_fed += loss * test_data_size[client]
                acc_test_fed += accuracy * test_data_size[client]

        # evaluate on all clients
        N2 = sum([v for k, v in testdata_size.items()])
        losstest_fed_all = losstest_fed_all / N2
        acctest_fed_all = acctest_fed_all / N2
        fed_metrics_all.loc[rnd] = [losstest_fed_all, acctest_fed_all]

        # metrics avg
        n1 = sum([v for k, v in train_data_size.items()])
        n2 = sum([v for k, v in test_data_size.items()])
        loss_train_fed = loss_train_fed / n1
        acc_train_fed = acc_train_fed / n1
        loss_test_fed = loss_test_fed / n2
        acc_test_fed = acc_test_fed / n2
        fed_metrics.loc[rnd] = [loss_train_fed, loss_test_fed, acc_train_fed, acc_test_fed]

        save_metrics()

        ## generalization test
        if args.new_test:
            losstest_fed_new, acctest_fed_new = 0, 0
            new_clients_datasize = [len(new_clients_test[i]) for i in newclient_list]
            new_clients_datasize = dict(zip(newclient_list, new_clients_datasize))
            n_new = sum([v for k, v in new_clients_datasize.items()])
            if not args.general_ensemble:
                for client in newclient_list:
                    generaltest = General_test(args, new_clients_train[client], model)
                    if args.method == 'fed' or args.method == 'fedprox':
                        generaltest.global_general(shared_layers_param)
                    if args.method == "fed_fac":
                        generaltest.FedFac_general(shared_layer_keys, shared_layers_param, partial_shared_layer_key, partial_conv_shared,
                           newtest_priv_para)
                    if args.method == 'fedper':
                        generaltest.FedPer_general(shared_layers_param, shared_layer_keys)
                    #---------------------------------
                    new_acc, new_loss = generaltest.general_test(rnd, new_clients_test[client], shared_layers_param)
                    loss_newtest.loc[rnd, client] = new_loss
                    acc_newtest.loc[rnd, client] = new_acc
                    losstest_fed_new += new_loss * new_clients_datasize[client]
                    acctest_fed_new += new_acc * new_clients_datasize[client]
                loss_new_fed = losstest_fed_new / n_new
                acc_new_fed = acctest_fed_new / n_new
                fed_metrics_new.loc[rnd] = [loss_new_fed, acc_new_fed]
                # save
                loss_newtest.to_excel(result_path + '/loss_newtest.xlsx')
                acc_newtest.to_excel(result_path + '/acc_newtest.xlsx')
                fed_metrics_new.to_excel(result_path + '/fed_metrics_new.xlsx')
            else:
                acc_avg_fed_new, acc_maj_fed_new = 0, 0
                net_local_list = [client_model[i] for i in client_sample]
                net_local_list = dict(zip(client_sample, net_local_list))
                GTE = General_test_ensemble(args, model)
                for client in newclient_list:
                    if args.method == 'lg':
                        acc_avg, loss, acc_maj = GTE.test_ensemble(net_local_list, new_clients_test[client], client_sample,
                       shared_layers_param)
                    if args.method == 'fed_fac':
                        acc_avg, loss, acc_maj = GTE.test_ensemble(net_local_list, new_clients_test[client], client_sample,
                                                                   shared_layers_param, partial_shared_layer_key, partial_conv_shared)
                    loss_newtest.loc[rnd, client] = loss
                    acc_avg_newtest.loc[rnd, client] = acc_avg
                    acc_maj_newtest.loc[rnd, client] = acc_maj
                    losstest_fed_new += loss * new_clients_datasize[client]
                    acc_avg_fed_new += acc_avg * new_clients_datasize[client]
                    acc_maj_fed_new += acc_maj * new_clients_datasize[client]
                loss_new_fed = losstest_fed_new / n_new
                acc_avg_fed_new = acc_avg_fed_new / n_new
                acc_maj_fed_new = acc_maj_fed_new / n_new
                fed_metrics_new.loc[rnd] = [loss_new_fed, acc_avg_fed_new, acc_maj_fed_new]
                # save
                loss_newtest.to_excel(result_path + '/loss_newtest.xlsx')
                acc_avg_newtest.to_excel(result_path + '/acc_avg_newtest.xlsx')
                acc_maj_newtest.to_excel(result_path + '/acc_maj_newtest.xlsx')
                fed_metrics_new.to_excel(result_path + '/fed_metrics_new.xlsx')

        # early_stop
        # if args.early_stop:
        #     score = acc_test_fed
        #     if best_score is None:
        #         best_score = score
        #         if args.partial_fed:
        #             save_checkpoint_p()
        #         else:
        #             save_checkpoint_s()
        #     elif score < best_score + delta:
        #         counter += 1
        #         print(f'earlystopping counter:{counter} out of {patience}')
        #         if counter >= patience:
        #             early_stop = True
        #     else:
        #         best_score = score
        #         if args.partial_fed:
        #             save_checkpoint_p()
        #         else:
        #             save_checkpoint_s()
        #         counter = 0
        # else:
        #     if args.save_model:
        #         if args.partial_fed:
        #             save_checkpoint_p()
        #         else:
        #             save_checkpoint_s()
        #
        # if args.early_stop:
        #     if early_stop == True:
        #         break
    if args.save_model:
        save_checkpoint()

if __name__ == '__main__':
    main()








