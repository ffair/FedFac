import copy
import os
import time
import random

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import numpy.linalg as LA

from utils.torch_utils import *


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients: List[Client]

    test_clients: List[Client]

    global_learners_ensemble: List[Learner]

    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    n_clients:

    n_learners:

    clients_weights:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    global_train_logger:

    global_test_logger:

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__
    mix

    update_clients

    update_test_clients

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            args,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            file_path=None,
            loss_train_df=None,
            loss_test_df=None,
            acc_train_df=None,
            acc_test_df=None,
            fed_metrics_df=None,
            loss_train_new_df=None,
            loss_test_new_df=None,
            acc_train_new_df=None,
            acc_test_new_df=None,
            fed_metrics_new_df = None
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.args = args
        self.file_path = file_path
        self.loss_train_df = loss_train_df
        self.loss_test_df = loss_test_df
        self.acc_train_df = acc_train_df
        self.acc_test_df = acc_test_df
        self.fed_metrics_df = fed_metrics_df

        self.loss_train_new_df = loss_train_new_df
        self.loss_test_new_df = loss_test_new_df
        self.acc_train_new_df = acc_train_new_df
        self.acc_test_new_df = acc_test_new_df
        self.fed_metrics_new_df = fed_metrics_new_df

        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.global_learners_ensemble = global_learners_ensemble
        self.device = self.global_learners_ensemble.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger

        self.model_dim = self.global_learners_ensemble.model_dim

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)
        self.n_learners = len(self.global_learners_ensemble)

        self.clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()
        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.c_round = 0

    def save_metrics(self):
        self.loss_train_df.to_excel(self.file_path + '/losstrain.xlsx')
        self.loss_test_df.to_excel(self.file_path + '/losstest.xlsx')
        self.acc_test_df.to_excel(self.file_path + '/acctest.xlsx')
        self.acc_train_df.to_excel(self.file_path + '/acctrain.xlsx')
        self.fed_metrics_df.to_excel(self.file_path + '/fedmetrics.xlsx')

    def save_metrics_g(self):
        self.loss_train_new_df.to_excel(self.file_path + '/losstrain_new.xlsx')
        self.loss_test_new_df.to_excel(self.file_path + '/losstest_new.xlsx')
        self.acc_test_new_df.to_excel(self.file_path + '/acctest_new.xlsx')
        self.acc_train_new_df.to_excel(self.file_path + '/acctrain_new.xlsx')
        self.fed_metrics_new_df.to_excel(self.file_path + '/fedmetrics_new.xlsx')


    @abstractmethod
    def mix(self, learners_model, learners_optim, learners_optim_lrsched):
        """

        Args:
            learners_model: [learner_model]
            clients_model: {[model.state_dict]}

        Returns:

        """
        pass

    @abstractmethod
    def update_clients(self, learners_model):
        pass

    def update_test_clients(self, learners_model):
        for client_id, client in enumerate(self.test_clients):
            for learner_id, learner in enumerate(client.learners_ensemble):
                shared_model = self.global_learners_ensemble[learner_id].model_state
                shared_model = {k: v.cpu() for k, v in shared_model.items()}
                learner.model_state = copy.deepcopy(shared_model)

        for client in self.test_clients:
            client.update_sample_weights(learners_model)
            client.update_learners_weights()

    def write_logs(self, learners_model):
        global_train_loss = 0.
        global_train_acc = 0.
        global_test_loss = 0.
        global_test_acc = 0.

        total_n_samples = 0
        total_n_test_samples = 0

        for client_id, client in enumerate(self.clients):
            train_loss, train_acc, test_loss, test_acc = client.write_logs(learners_model)
            if client.client_name is not None:
                client_name = client.client_name
            else:
                client_name = client_id
            self.loss_train_df.loc[self.c_round, client_name] = train_loss
            self.loss_test_df.loc[self.c_round, client_name] = test_loss
            self.acc_train_df.loc[self.c_round, client_name] = train_acc
            self.acc_test_df.loc[self.c_round, client_name] = test_acc

            global_train_loss += train_loss * client.n_train_samples
            global_train_acc += train_acc * client.n_train_samples
            global_test_loss += test_loss * client.n_test_samples
            global_test_acc += test_acc * client.n_test_samples

            total_n_samples += client.n_train_samples
            total_n_test_samples += client.n_test_samples

        global_train_loss /= total_n_samples
        global_test_loss /= total_n_test_samples
        global_train_acc /= total_n_samples
        global_test_acc /= total_n_test_samples
        self.fed_metrics_df.loc[self.c_round] = [global_train_loss, global_test_loss, global_train_acc, global_test_acc]
        self.save_metrics()

        # clients for new test
        if self.args.new_test:
            self.update_test_clients(learners_model)
            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.
            total_n_samples = 0
            total_n_test_samples = 0
            for client_id, client in enumerate(self.test_clients):
                train_loss, train_acc, test_loss, test_acc = client.write_logs(learners_model)
                if client.client_name is not None:
                    client_name = client.client_name
                else:
                    client_name = client_id
                self.loss_train_new_df.loc[self.c_round, client_name] = train_loss
                self.loss_test_new_df.loc[self.c_round, client_name] = test_loss
                self.acc_train_new_df.loc[self.c_round, client_name] = train_acc
                self.acc_test_new_df.loc[self.c_round, client_name] = test_acc

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples
            self.fed_metrics_new_df.loc[self.c_round] = [global_train_loss, global_test_loss, global_train_acc, global_test_acc]
            self.save_metrics_g()


    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            save_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            torch.save(learner.model.state_dict(), save_path)

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            save_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            for client_id, client in enumerate(clients):
                weights[client_id] = client.learners_ensemble.learners_weights

            np.save(save_path, weights)

    def load_state(self, dir_path):
        """
        load the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         from a `.pt` file, and `learners_weights` for each client in `self.clients` from numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            chkpts_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            learner.model.load_state_dict(torch.load(chkpts_path))

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            chkpts_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            weights = np.load(chkpts_path)

            for client_id, client in enumerate(clients):
                client.learners_ensemble.learners_weights = weights[client_id]

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients = \
                self.rng.choices(
                    population=self.clients,
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients = self.rng.sample(self.clients, k=self.n_clients_per_round)


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self, learners_model, learners_optim, learners_optim_lrsched):
        self.sample_clients()

        for client_id, client in enumerate(self.sampled_clients):
            _ = client.step(learners_model, learners_optim, learners_optim_lrsched)

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs(learners_model)
        # return clients_model

    def update_clients(self, learners_model):
        pass


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """
    def mix(self, learners_model, learners_optim, learners_optim_lrsched):
        self.sample_clients()

        # for client in self.sampled_clients:
        for client_id, client in enumerate(self.sampled_clients):
            _ = client.step(learners_model, learners_optim, learners_optim_lrsched)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            # average all clients' parameters.--> average parameters of clients which participate training
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            # learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]

            average_learners(learners, learner, weights=self.clients_weights)

        # assign the updated model to all clients
        self.update_clients(learners_model)

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs(learners_model)


    def update_clients(self, learners_model):
        for client_id, client in enumerate(self.clients):
            for learner_id, learner in enumerate(client.learners_ensemble):
                shared_model = self.global_learners_ensemble[learner_id].model_state
                shared_model = {k: v.cpu() for k, v in shared_model.items()}
                learner.model_state = copy.deepcopy(shared_model)






