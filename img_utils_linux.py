# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:45:27 2021

@author: THY
"""

import math
from scipy import stats
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import string
from tqdm import tqdm
from torch.nn import init
from factor_analyzer import FactorAnalyzer
import numpy.linalg as nlg
from sklearn import preprocessing
import numpy as np
import pandas as pd
import copy
from collections import Counter
import torch.utils.data as Data
from torchvision import datasets, transforms
import os


class NextCharacterLSTM(nn.Module):
    def __init__(self, args, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if (torch.cuda.is_available()) & (args.device == 'gpu') else 'cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.fc_drop = nn.Dropout(p=0.5)
        self.encoder = nn.Embedding(input_size, embed_size)
        self.rnn = \
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, input_):
        # Set initial states
        h0 = torch.zeros(self.n_layers, input_.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.n_layers, input_.size(0), self.hidden_size).cuda()
        # encoded = self.encoder(input_).to(self.device)
        encoded = self.encoder(input_)
        if self.args.device == 'gpu':
            h0 = h0.cuda()
            c0 = c0.cuda()
            encoded = encoded.cuda()

        # encoded = self.encoder(input_)
        # output, _ = self.rnn(encoded)
        output, _ = self.rnn(encoded, (h0, c0))
        output = F.relu(self.fc1(output))
        output = self.fc_drop(output)
        output = self.fc2(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        num_channels = args.num_channels.split(",")
        num_channels = [int(item) for item in num_channels]
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1 = nn.Conv2d(1, num_channels[0], kernel_size=5)
        #.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(num_channels[0], num_channels[1], kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p = 0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc_drop = nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Sequential(
            nn.Linear(512, args.num_classes),
            torch.nn.Softmax(dim=1)
        )
        #.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = torch.flatten(x, start_dim= 1)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = self.fc2(x)
        return x


class CharacterDataset(Dataset):
    def __init__(self, file_path, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input
        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences
        """
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len

        with open(file_path, 'r') as f:
            self.text = f.read()

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx+self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx+1:idx+self.chunk_len+1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class Shakespeare_util:
    def __init__(self, args, ):
        self.args = args
        self.device = torch.device('cuda' if (torch.cuda.is_available()) & (self.args.device == 'gpu') else 'cpu')
        self.root_path = self.args.shakespeare_root_path
        self.SHAKESPEARE_CONFIG = {
            "input_size": len(string.printable),
            "embed_size": 8,
            "hidden_size": 256,
            "output_size": len(string.printable),
            "n_layers": 2,
            "chunk_len": 80
        }
        self.CHARACTERS_WEIGHTS = {
            '\n': 0.43795308843799086,
            ' ': 0.042500849608091536,
            ',': 0.6559597911540539,
            '.': 0.6987226398690805,
            'I': 0.9777491725556848,
            'a': 0.2226022051965085,
            'c': 0.813311655455682,
            'd': 0.4071860494572223,
            'e': 0.13455606165058104,
            'f': 0.7908671114133974,
            'g': 0.9532922255751889,
            'h': 0.2496906467588955,
            'i': 0.27444893060347214,
            'l': 0.37296488139109546,
            'm': 0.569937324017103,
            'n': 0.2520734570378263,
            'o': 0.1934141300462555,
            'r': 0.26035705948768273,
            's': 0.2534775933879391,
            't': 0.1876471355731429,
            'u': 0.47430062920373184,
            'w': 0.7470615815733715,
            'y': 0.6388302610200002
        }
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=self.device)
        for character in self.CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = self.CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8
        self.criterion = nn.CrossEntropyLoss(reduction="sum", weight=labels_weight).to(self.device)

    def get_dataloader(self):
        train_iterators, test_iterators = {}, {}
        for task_id, task_dir in enumerate(tqdm(os.listdir(self.root_path))):
            task_data_path = os.path.join(self.root_path, task_dir)

            dataset1 = CharacterDataset(file_path=os.path.join(task_data_path, "train.txt"),
                                       chunk_len=self.SHAKESPEARE_CONFIG["chunk_len"])

            dataset2 = CharacterDataset(file_path=os.path.join(task_data_path, "test.txt"),
                                       chunk_len=self.SHAKESPEARE_CONFIG["chunk_len"])
            if (len(dataset1) > 0) and (len(dataset2)>0):
                train_iterators[str(task_id)] = dataset1
                test_iterators[str(task_id)] = dataset2
        return train_iterators, test_iterators

    def get_model(self):
        model = \
            NextCharacterLSTM(
                args = self.args,
                input_size=self.SHAKESPEARE_CONFIG["input_size"],
                embed_size=self.SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=self.SHAKESPEARE_CONFIG["hidden_size"],
                output_size=self.SHAKESPEARE_CONFIG["output_size"],
                n_layers=self.SHAKESPEARE_CONFIG["n_layers"]
            )
        return model


class LocalUpdate(object):
    def __init__(self, args, train_data, net):
        self.args = args
        self.net = net
        self.device = torch.device('cuda' if (torch.cuda.is_available()) & (self.args.device == 'gpu') else 'cpu')
        if self.args.dataset == 'shakespeare':
            self.shakespeare_util = Shakespeare_util(self.args)
            self.loss_func = self.shakespeare_util.criterion
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='sum').to(self.device)
        self.train_data = train_data

    def train(self, rnd=None, global_param = None, shared_layer_keys = None, shared_layers_param = None):
        self.net.train()
        self.net.zero_grad()
        if self.args.optimizer == 'adam': # lr=0.001
            optimizer = torch.optim.Adam(self.net.parameters(), lr = self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.args.weight_decay,
                                         amsgrad=False)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.net.parameters(), lr= self.args.lr, momentum=0.9, dampening=0, weight_decay=self.args.weight_decay,
                                        nesterov=False)
        # if self.args.lr_scheduler:
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #                                                       lr_lambda=lambda rnd: self.args.lmbda ** (rnd // self.args.lr_drop))
        # 手动调整学习率
        if self.args.lr_scheduler:
            if rnd % self.args.lr_drop == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= self.args.lmbda ** (rnd // self.args.lr_drop)
                    print('rnd{}: lr{}'.format(rnd, optimizer.state_dict()['param_groups'][0]['lr']))

        epoch_loss, epoch_acc = [], []
        for epoch in range(self.args.local_ep):
            train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True)
            batch_loss = []
            correct = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                if self.args.dataset == 'sent140':
                    images, labels = images[0].permute(1, 0), images[1]
                else:
                    images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                if self.args.fedprox:
                    if rnd>0:
                        w_diff = torch.tensor(0., device=self.device)
                        for w, w_t in zip(shared_layers_param.values(), self.net.parameters()):
                            w_diff += torch.pow(torch.norm(w.to(self.device) - w_t.to(self.device)), 2)
                        loss += self.args.mu / 2. * w_diff

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            ## lr update
            # if (self.args.optimizer == 'sgd') and (self.args.lr_scheduler):
            #     scheduler.step()
            if self.args.dataset == 'shakespeare':
                chunk_len = self.shakespeare_util.SHAKESPEARE_CONFIG['chunk_len']
                epoch_loss.append(sum(batch_loss) / chunk_len / len(train_loader.dataset))
                accuracy = correct / chunk_len / len(train_loader.dataset)
            else:
                epoch_loss.append(sum(batch_loss) / len(train_loader.dataset))
                accuracy = correct / len(train_loader.dataset)
            epoch_acc.append(accuracy)
        a = sum(epoch_acc) / len(epoch_acc)
        net_weight = self.net.state_dict()
        net_weight = {k: v.cpu() for k, v in net_weight.items()}
        #
        if self.args.save_model:
            optim = copy.deepcopy(optimizer)
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
            return net_weight, sum(epoch_loss) / len(epoch_loss), a.item(), optim.state_dict()
        else:
            return net_weight, sum(epoch_loss) / len(epoch_loss), a.item()


def test_img(rnd, net_g, dataset, args, shared_layers_param, return_probs=False, global_param = None):
    device = torch.device('cuda' if (torch.cuda.is_available()) & (args.device == 'gpu') else 'cpu')
    net_g.eval()
    if args.dataset == 'shakespeare':
        shakespeare_util = Shakespeare_util(args)
        loss_func = shakespeare_util.criterion
    else:
        loss_func = nn.CrossEntropyLoss(reduction='sum').to(device)
    if args.dataset == 'shakespeare':
        shakespeare_util = Shakespeare_util(args)
        loss_func = shakespeare_util.criterion

    # testing
    test_loss = 0
    correct = 0
    probs = []
    labels = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            labels += list(target.numpy())
            if args.dataset == 'sent140':
                data, target = data[0].permute(1, 0), data[1]
            else:
                data, target = data.to(device), target.to(device)
            log_probs = net_g(data)
            probs.append(log_probs)
            # sum up batch loss
            if args.fedprox:
                loss = loss_func(log_probs, target)
                if rnd > 0:
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(shared_layers_param.values(), net_g.parameters()):
                        w_diff += torch.pow(torch.norm(w.to(device) - w_t.to(device)), 2)
                    loss += args.mu / 2. * w_diff
                    test_loss += loss.item()
            else:
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        if args.dataset == 'shakespeare':
            chunk_len = shakespeare_util.SHAKESPEARE_CONFIG['chunk_len']
            test_loss /= len(data_loader.dataset)
            test_loss /= chunk_len
            accuracy = correct / len(data_loader.dataset) / chunk_len
        else:
            test_loss /= len(data_loader.dataset)
            accuracy = correct / len(data_loader.dataset)
    if return_probs:
        return accuracy.item(), test_loss, torch.cat(probs), labels
    return accuracy.item(), test_loss


class General_test(object):
    def __init__(self, args, train_data, net):
        self.args = args
        self.net = net
        self.train_data = train_data # new_clients_train[client]
        self.device = torch.device('cuda' if (torch.cuda.is_available()) & (args.device == 'gpu') else 'cpu')
        if self.args.dataset == 'shakespeare':
            self.shakespeare_util = Shakespeare_util(self.args)
            self.loss_func = self.shakespeare_util.criterion
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='sum').to(self.device)

    def load_global_param(self, shared_layers_param):
        model_dict = copy.deepcopy(self.net.state_dict())
        model_dict.update(copy.deepcopy(shared_layers_param))
        self.net.load_state_dict(model_dict)

    def global_general(self, shared_layers_param):
        self.load_global_param(shared_layers_param)

    def FedFac_general(self, shared_layer_keys, shared_layers_param, partial_shared_layer_key, partial_conv_shared,
                       newtest_priv_para):
        if self.args.fac_newtest_priv == "train-fix":
            # update private split layer parameter
            localupdate = LocalUpdate(args=self.args, train_data=self.train_data, net=self.net)
            localupdate.train(shared_layer_keys = shared_layer_keys, shared_layers_param = shared_layers_param)

        self.load_global_param(shared_layers_param)
        for key in partial_shared_layer_key:
            partial_fed_avg = Partial_fed_avg(self.args, None, None, key)
            # update shared split layer parameter
            partial_fed_avg.partial_param_update(self.net, self.net.state_dict(), partial_conv_shared[key])
            # update private split layer parameter
            if self.args.fac_newtest_priv == "Avg":
                partial_fed_avg.partial_param_update(self.net, self.net.state_dict(), newtest_priv_para[key])
            if self.args.fac_newtest_priv == "fix-tune":
                self.net.train()
                self.net.zero_grad()
                # froze global shared parameters
                for name, param in self.net.named_parameters():
                    if name in shared_layer_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, betas=(0.9, 0.999),
                                             eps=1e-08, weight_decay=self.args.weight_decay,
                                             amsgrad=False)
                for epoch in range(self.args.local_ep):
                    train_loader = torch.utils.data.DataLoader(self.train_data,
                                                               batch_size=self.args.batch_size,
                                                               shuffle=True)
                    for batch_idx, (images, labels) in enumerate(train_loader):
                        images, labels = images.to(self.device), labels.to(self.device)
                        self.net.zero_grad()
                        log_probs = self.net(images)
                        loss = self.loss_func(log_probs, labels)
                        loss.backward()
                        optimizer.step()
                        partial_fed_avg.partial_param_update(self.net, self.net.state_dict(), partial_conv_shared[key])

    def FedPer_general(self, shared_layers_param, shared_layer_keys):
        # load, fix and tune
        self.load_global_param(shared_layers_param)
        self.net.train()
        self.net.zero_grad()
        # froze global shared parameters
        for name, param in self.net.named_parameters():
            if name in shared_layer_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=self.args.weight_decay,
                                     amsgrad=False)
        for epoch in range(self.args.local_ep):
            train_loader = torch.utils.data.DataLoader(self.train_data,
                                                       batch_size=self.args.batch_size,
                                                       shuffle=True)
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

    def general_test(self, rnd, dataset, shared_layers_param):
        acc, loss = test_img(rnd = rnd, net_g = self.net, dataset = dataset, args = self.args, shared_layers_param = shared_layers_param)
        return acc, loss


class General_test_ensemble(object):
    def __init__(self, args, net):
        self.args = args
        self.net = net
        self.device = torch.device('cuda' if (torch.cuda.is_available()) & (args.device == 'gpu') else 'cpu')
        if self.args.dataset == 'shakespeare':
            self.shakespeare_util = Shakespeare_util(self.args)
            self.loss_func = self.shakespeare_util.criterion
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='sum').to(self.device)
    def load_global_param(self, local_model, shared_layers_param):
        model_dict = local_model
        model_dict.update(copy.deepcopy(shared_layers_param))
        self.net.load_state_dict(model_dict)

    def load_fac_param(self, partial_shared_layer_key, partial_conv_shared,shared_layers_param, local_model):
        for key in partial_shared_layer_key:
            partial_fed_avg = Partial_fed_avg(self.args, None, None, key)
            # update shared split layer parameter
            partial_fed_avg.partial_param_update(self.net, local_model, partial_conv_shared[key])
        model_dict = self.net.state_dict()
        model_dict.update(copy.deepcopy(shared_layers_param))
        self.net.load_state_dict(model_dict)

    def test_ensemble(self, net_local_list, dataset_test, client_sample, shared_layers_param, partial_shared_layer_key = None,
                      partial_conv_shared =None):
        probs_all = []
        preds_all = []
        for client in client_sample:
            local_model = net_local_list[client]
            if self.args.method == 'lg':
                self.load_global_param(local_model, shared_layers_param)
            if self.args.method == "fed_fac":
                self.load_fac_param(partial_shared_layer_key, partial_conv_shared, shared_layers_param, local_model)
            acc, loss, probs, labels = test_img(rnd = None, net_g=self.net, dataset=dataset_test, args=self.args, shared_layers_param=shared_layers_param,
                     return_probs=True)
            probs_all.append(probs.detach())

            preds = probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
            preds_all.append(preds)

        # labels = np.array(dataset_test[1])
        labels = np.array(labels)
        preds_probs = torch.mean(torch.stack(probs_all), dim=0)

        # ensemble (avg) metrics
        preds_avg = preds_probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        loss_test = self.loss_func(preds_probs, torch.tensor(labels).to(self.device)).item()
        acc_test_avg = (preds_avg == labels).mean()

        # ensemble (maj)
        preds_all = np.array(preds_all).T
        preds_maj = stats.mode(preds_all, axis=1)[0].reshape(-1)
        acc_test_maj = (preds_maj == labels).mean()

        return acc_test_avg, loss_test, acc_test_maj



class Partial_fed_avg():
    """
        把client对应序号的filter拉直后拼接，组成client*filter的二维矩阵，对该矩阵做因子分析，得到因子载荷矩阵。
        求每个filter的因子载荷的平方和，认为平方和大于平均平方和的filter主要提取公共信息，对该部分filter进行client-wise平均
        """
    def __init__(self, args, local_params, train_data_size, partial_shared_layer_key):
        self.args = args
        self.device = torch.device('cuda' if (torch.cuda.is_available()) & (self.args.device == 'gpu') else 'cpu')
        self.local_params = local_params
        self.train_data_size = train_data_size
        self.partial_shared_layer_key = partial_shared_layer_key

    def my_func(self, x):
        return sum([np.square(i) for i in x])

    def partial_para_avg(self, priv_indx):
        """
        input：
            priv_indx：the index of elements need to be averaged
        output：partial_conv_shared -->dict{index:param}
            
        """
        partial_conv_shared = {}
        total_n = sum([v for k,v in self.train_data_size.items()])
        for indx in priv_indx:
            share_p = []
            for i in self.local_params.keys():
                p = self.local_params[i][self.partial_shared_layer_key]
                if self.args.device == 'gpu':
                    p = p.cpu().detach().numpy()
                else:
                    p = p.numpy()
                share_p.append(p[indx,] * self.train_data_size[i] / total_n)
            share_p = np.array(share_p).sum(axis = 0)
            partial_conv_shared[indx] = share_p
        return partial_conv_shared

    def partial_param_update(self, model, local_model, partial_conv_shared):
        """
        Args:
            model:
            local_model: local model.state_dict()
            partial_conv_shared: cpu上

        Returns:

        """
        model_dict = copy.deepcopy(local_model)
        p = model_dict[self.partial_shared_layer_key]
        if self.args.device == 'gpu':
            p = p.cpu().detach().numpy()
        else:
            p = p.numpy()
        for indx in partial_conv_shared.keys():
            p[indx,] = partial_conv_shared[indx]
        p = torch.from_numpy(p).to(self.device)
        model_dict[self.partial_shared_layer_key] = p
        model.load_state_dict(model_dict)

    def partial_shared_removed(self, model, local_model, partial_conv_shared):
        """replace shared parameters with random value"""
        model_dict = copy.deepcopy(local_model)
        p = model_dict[self.partial_shared_layer_key]
        if self.args.device == 'gpu':
            p = p.cpu().detach().numpy()
        else:
            p = p.numpy()
        if len(p[0,].shape) > 2:
            for indx in partial_conv_shared.keys():
                p[indx,] = np.random.randn(p[indx,].shape[0], p[indx,].shape[1], p[indx,].shape[2])
        if len(p[0,].shape) == 2:
            for indx in partial_conv_shared.keys():
                p[indx,] = np.random.randn(p[indx,].shape[0], p[indx,].shape[1])

        p = torch.from_numpy(p)
        model_dict[self.partial_shared_layer_key] = p
        model.load_state_dict(model_dict)

    def private_removed(self, model, local_model, partial_conv_shared):
        """replace private parameters with random value"""
        model_dict = copy.deepcopy(local_model)
        p = model_dict[self.partial_shared_layer_key]
        if self.args.device == 'gpu':
            p = p.cpu().detach().numpy()
        else:
            p = p.numpy()
        for indx in partial_conv_shared.keys():
            p[indx,] = partial_conv_shared[indx]

        private_indx = list(set(range(len(p))) - set(partial_conv_shared.keys()))
        if len(p[0,].shape) > 2:
            for indx in private_indx:
                p[indx,] = np.random.randn(p[indx,].shape[0], p[indx,].shape[1], p[indx,].shape[2])
        if len(p[0,].shape) == 2:
            for indx in private_indx:
                p[indx,] = np.random.randn(p[indx,].shape[0], p[indx,].shape[1])

        p = torch.from_numpy(p)
        model_dict[self.partial_shared_layer_key] = p
        model.load_state_dict(model_dict)

    def get_s_p_indx(self, fac_type):
        """return index of shared filters and private filters"""
        conv_p = []
        for i in self.local_params.keys():
            p = self.local_params[i][self.partial_shared_layer_key]
            if fac_type == 'conv':
                p = torch.flatten(p, start_dim = 1)
            if self.args.device == 'gpu':
                p = p.cpu().detach().numpy()
            else:
                p = p.numpy()
            conv_p.append(p)
        conv_p = np.array(conv_p)
        conv_p = torch.from_numpy(conv_p)
        a = conv_p.permute((1, 0,2))
        a = torch.flatten(a, start_dim= 1)
        a = a.numpy().T
        total_indx = list(range(a.shape[1]))
        a = preprocessing.scale(a)
        ## 特征分解
        if self.args.eig_num == None:
            corr = np.corrcoef(a.T)
            eig_value, eig_vec = nlg.eig(corr)
            eig_value = [item.real for item in eig_value]
            # 对特征值按降序排列
            eig_indx = np.argsort(eig_value)[::-1]
            eig_vec1 = eig_vec[:,eig_indx]
            eig_num = 0
            eig_sum = 0
            for m in eig_indx:
                eig_sum += eig_value[m]
                if eig_sum / sum(eig_value) >= self.args.cov_thrhd:
                    break
                eig_num += 1
            # print(eig_num)
        else:
            eig_num = self.args.eig_num
        if self.args.direct_eigDecom:
            loadings = eig_vec1[:,:eig_num]
            fac_load_sum2 = np.apply_along_axis(self.my_func, 1, loadings)
        else:
            fa = FactorAnalyzer(eig_num, rotation = None)
            fa.fit(a)
            fac_load_sum2 = np.apply_along_axis(self.my_func, 1, fa.loadings_)
        if self.args.threshold_p:
            aa1 = pd.Series(fac_load_sum2).map(lambda x: 0 if x < np.percentile(fac_load_sum2, self.args.threshold) else 1)
        elif self.args.given_threshold:
            aa1 = pd.Series(fac_load_sum2).map(
                lambda x: 0 if x < self.args.given_threshold_v else 1)
        else:
            aa1 = pd.Series(fac_load_sum2).map(lambda x: 0 if x < np.mean(fac_load_sum2) else 1)
        priv_indx = [i for i, x in enumerate(aa1) if x == 0]
        share_indx = [i for i, x in enumerate(aa1) if x == 1]
        return priv_indx, share_indx
    


def personalize_FedAvg(client_sample, local_params, train_data_size, shared_layer_keys):
    """
    Parameters
    ----------
    local_params : dict of model.state_dict()
        list of local model's parameters.
    train_data_size : dict
        number of samples on each client.
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    ave_param : dict
        the averaged parameters of the shared model layers.

    """
    user = client_sample
    ave_param = {}
    for key in shared_layer_keys:
        w_avg = copy.deepcopy(local_params[user[0]][key]) * train_data_size[user[0]]
        sample_num = train_data_size[user[0]]
        for i in user[1:]:
            w_avg += local_params[i][key] * train_data_size[i]
            sample_num += train_data_size[i]
        ave_param[key] = w_avg / sample_num
    return ave_param




class Data_Process():
    def __init__(self, args, ):
        self.args = args
        if self.args.dataset == 'mnist':
            self.cached_file_pfl = self.args.mnist_cached_file_pfl
            self.raw_data = self.args.mnist_rawdata
        if self.args.dataset == 'cifar':
            self.cached_file_pfl = self.args.cifar_cached_file_pfl
            self.raw_data = self.args.cifar_rawdata

    def fetch_data(self):
        path = ''
        if self.args.dataset == 'mnist':
            if self.args.devide_by_label:
                path = self.cached_file_pfl + '_{}clients'.format(self.args.num_classes) + '/mnist_torchdata'
            else:
                if self.args.sampling == 'dirichlet':
                    path = self.cached_file_pfl + '/dirichlet_parameter_{}'.format(self.args.dirichlet_parameter) + '/torchdata'
        if self.args.dataset == 'cifar':
            if self.args.sampling == 'dirichlet':
                path = self.cached_file_pfl + '/dirichlet_parameter_{}'.format(self.args.dirichlet_parameter)

        if os.path.exists(path):
            data_load = torch.load(path)
            train_clients = data_load['traindataset']
            test_clients = data_load['testdataset']
        else:
            if self.args.devide_by_label:
                train_clients, test_clients = self.get_10_clients()
            else:
                train_clients, test_clients = self.get_data_ready()
        return train_clients, test_clients

    def get_data_ready(self):
        if self.args.dataset == 'mnist':
            raw_train = datasets.MNIST(root = self.raw_data, train = True, transform = transforms.ToTensor(), download = True)
            raw_test = datasets.MNIST(root = self.raw_data, train = False, transform = transforms.ToTensor(), download = True)
        if self.args.dataset == 'cifar':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=mean),
            ])
            raw_train = datasets.CIFAR10(root= self.raw_data, train=True, transform=transform_train, download=True)
            raw_test = datasets.CIFAR10(root= self.raw_data, train=False, transform=transform_test, download=True)

        train_x, train_y = [],[]
        test_x, test_y = [], []
        for i in range(len(raw_train)):
            a,b = raw_train[i]
            train_x.append(a.numpy())
            train_y.append(b)
        for i in range(len(raw_test)):
            a,b = raw_test[i]
            test_x.append(a.numpy())
            test_y.append(b)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        NUM_EXAMPLES_PER_CLIENT = round(len(train_x) / self.args.num_clients)
        TEST_SAMPLES_PER_CLIENT = round(len(test_x) / self.args.num_clients)
        NUM_CLASSES = self.args.num_classes
        NUM_CLIENTS = self.args.num_clients

        def dirichlet_sample():
            train_example_indices = []
            test_indices = []
            for k in range(NUM_CLASSES):
                train_label_k = np.where(train_y == k)[0]
                np.random.shuffle(train_label_k)
                train_example_indices.append(train_label_k)
                test_label_k = np.where(test_y == k)[0]
                np.random.shuffle(test_label_k)
                test_indices.append(test_label_k)

            train_example_indices = np.array(train_example_indices)
            test_indices = np.array(test_indices)

            count_train = [len(item) for item in train_example_indices]
            count_test = [len(item)  for item in test_indices]

            ##
            train_multinomial_vals = []
            test_multinomial_vals = []

            # Each client has a multinomial distribution over classes drawn from a Dirichlet.
            np.random.seed(108)
            for i in range(NUM_CLIENTS):
                proportion = np.random.dirichlet(self.args.dirichlet_parameter * np.ones(NUM_CLASSES,))
                train_multinomial_vals.append(proportion)
                test_multinomial_vals.append(proportion)

            train_multinomial_vals = np.array(train_multinomial_vals, dtype = np.float64)
            test_multinomial_vals = np.array(test_multinomial_vals,dtype = np.float64)

            train_client_samples = [[] for _ in range(NUM_CLIENTS)]
            test_client_samples = [[] for _ in range(NUM_CLIENTS)]
            train_count = np.zeros(NUM_CLASSES).astype(int)
            test_count = np.zeros(NUM_CLASSES).astype(int)

            ##
            for k in range(NUM_CLIENTS):
                if k < len(train_multinomial_vals):
                    for i in range(NUM_EXAMPLES_PER_CLIENT):
                        if train_multinomial_vals[k, :].sum() ==0.:
                            break
                        sampled_label = np.argwhere(np.random.multinomial(1, train_multinomial_vals[k, :]) == 1)[0][0]
                        train_client_samples[k].append(
                          train_example_indices[sampled_label][train_count[sampled_label]])
                        train_count[sampled_label] += 1
                        if train_count[sampled_label] == count_train[sampled_label]:
                            train_multinomial_vals[:, sampled_label] = 0.
                            a = train_multinomial_vals.sum(axis=1)[:, None]
                            a[a==0.] = 1.
                            train_multinomial_vals = train_multinomial_vals / a
                            train_multinomial_vals = train_multinomial_vals[~np.isnan(train_multinomial_vals).any(axis = 1)]

                    for i in range(TEST_SAMPLES_PER_CLIENT):
                        if test_multinomial_vals[k, :].sum() == 0.:
                            break
                        sampled_label = np.argwhere(np.random.multinomial(1, test_multinomial_vals[k, :]) == 1)[0][0]
                        test_client_samples[k].append(test_indices[sampled_label][test_count[sampled_label]])
                        test_count[sampled_label] += 1
                        if test_count[sampled_label] == count_test[sampled_label]:
                            test_multinomial_vals[:, sampled_label] = 0.
                            a = test_multinomial_vals.sum(axis=1)[:, None]
                            a[a==0.] = 1.
                            test_multinomial_vals = test_multinomial_vals / a  # [:, None]是为了归一化
                            test_multinomial_vals = test_multinomial_vals[~np.isnan(test_multinomial_vals).any(axis = 1)] # 去掉含nan的行

            train_client_samples = list(filter(None, train_client_samples))
            test_client_samples = list(filter(None, test_client_samples))
            return train_client_samples, test_client_samples

        if self.args.sampling == 'dirichlet':
            train_client_samples, test_client_samples = dirichlet_sample()

        train_clients, test_clients = '',''
        if self.args.sampling != 'mix_dirichlet':
            train_clients = {}
            test_clients = {}
            NUM_CLIENTS = min(len(train_client_samples), len(test_client_samples))
            for i in range(NUM_CLIENTS):
                client_name = str(i)
                x_train = train_x[np.array(train_client_samples[i])]
                y_train = train_y[np.array(train_client_samples[i])].squeeze()
                train_data = Data.TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train, dtype=torch.long))
                train_clients[client_name] = train_data

                x_test = test_x[np.array(test_client_samples[i])]
                y_test = test_y[np.array(test_client_samples[i])].squeeze()
                test_data = Data.TensorDataset(torch.from_numpy(x_test), torch.tensor(y_test, dtype=torch.long))
                test_clients[client_name] = test_data
            # 计算各client中 0-9 的比例，以显示 non-iid 的程度
            number_ratio = pd.DataFrame(columns=[str(i) for i in range(10)])
            for j in range(NUM_CLIENTS):
                a1 = Counter(train_y[i] for i in train_client_samples[j])
                number_ratio.loc[j] = [a1[0], a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a1[8], a1[9]]

            number_ratio_test = pd.DataFrame(columns=[str(i) for i in range(10)])
            for j in range(NUM_CLIENTS):
                a1 = Counter(test_y[i] for i in test_client_samples[j])
                number_ratio_test.loc[j] = [a1[0], a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a1[8], a1[9]]

        # 保存数据
        path_0 = ''
        if self.args.sampling == 'dirichlet':
            path_0 = self.cached_file_pfl + '/dirichlet_parameter_{}'.format(self.args.dirichlet_parameter)

        if not os.path.exists(path_0):
            os.makedirs(path_0)
        path = path_0 + '/torchdata'
        torch.save({'traindataset': train_clients, 'testdataset': test_clients}, path)

        if self.args.sampling == 'dirichlet':
            path1 = path_0 + '/dirichlet_parameter{}.xlsx'.format(self.args.dirichlet_parameter)
            path2 = path_0 + '/dirichlet_parameter{}_test.xlsx'.format(self.args.dirichlet_parameter)
        number_ratio.to_excel(path1, index=False)
        number_ratio_test.to_excel(path2, index=False)
        return train_clients, test_clients
    
    def get_10_clients(self):
        NUM_CLASSES = self.args.num_classes

        mnist_train = datasets.MNIST(root = self.raw_data, train = True, transform = transforms.ToTensor(), download = False)
        mnist_test = datasets.MNIST(root = self.raw_data, train = False, transform = transforms.ToTensor(), download = False)
        
        mnist_train_x, mnist_train_y = [],[]
        mnist_test_x, mnist_test_y = [], []
        for i in range(len(mnist_train)):
            a,b = mnist_train[i]
            mnist_train_x.append(a.numpy())
            mnist_train_y.append(b)
        for i in range(len(mnist_test)):
            a,b = mnist_test[i]
            mnist_test_x.append(a.numpy())
            mnist_test_y.append(b)
        mnist_train_x = np.array(mnist_train_x)
        mnist_train_y = np.array(mnist_train_y)
        mnist_test_x = np.array(mnist_test_x)
        mnist_test_y = np.array(mnist_test_y)

        train_example_indices = []
        test_indices = []
        for k in range(NUM_CLASSES):
            train_label_k = np.where(mnist_train_y == k)[0]
            np.random.shuffle(train_label_k)
            m = round(len(train_label_k)/10)
            for ki in range(9):
                train_example_indices.append(train_label_k[ki*m:(ki+1)*m])
            train_example_indices.append(train_label_k[9 * m:])

            test_label_k = np.where(mnist_test_y == k)[0]
            np.random.shuffle(test_label_k)
            m = round(len(test_label_k) / 10)
            for ki in range(9):
                test_indices.append(test_label_k[ki * m:(ki + 1) * m])
            test_indices.append(test_label_k[9 * m:])

        train_example_indices = np.array(train_example_indices)
        test_indices = np.array(test_indices)
        
        train_clients = {}
        test_clients = {}
        for i in range(NUM_CLASSES*10):
            client_name = str(i)
            x_train = mnist_train_x[np.array(train_example_indices[i,])]
            y_train = mnist_train_y[np.array(train_example_indices[i,])].squeeze()
            train_data = Data.TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train, dtype=torch.long))
            train_clients[client_name] = train_data
        
            x_test = mnist_test_x[np.array(test_indices[i,])]
            y_test = mnist_test_y[np.array(test_indices[i,])].squeeze()
            test_data = Data.TensorDataset(torch.from_numpy(x_test), torch.tensor(y_test, dtype=torch.long))
            test_clients[client_name] = test_data

        path_0 = self.cached_file_pfl + '_clients{}'.format(self.args.num_classes)
        if not os.path.exists(path_0):
            os.makedirs(path_0)
        path = path_0 + '/mnist_torchdata'
        torch.save({'traindataset': train_clients, 'testdataset': test_clients}, path)
        number_ratio = pd.DataFrame(columns=[str(i) for i in range(NUM_CLASSES)])
        for j in range(NUM_CLASSES*10):
            a1 = Counter(mnist_train_y[i] for i in train_example_indices[j])
            number_ratio.loc[j] = [a1[0], a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a1[8], a1[9]]

        path = path_0 + '/summary.xlsx'
        number_ratio.to_excel(path, index=False)
        return train_clients, test_clients














